"""
BIRD-SQL 版本的 DeepSeek Relational Transformer (支持 CoT 生成)
- 移除了 RelBench 依赖
- 移除了分类头，改为 Causal LM 生成头
- 增加了 generate() 方法支持思维链 (CoT)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from torch_frame import stype
from task_type import TaskType  # 使用自定义 TaskType
import types

# 默认使用的文本编码维度 (DeepSeek-R1-Distill-Qwen-1.5B 为 1536, MiniLM 为 384, 7B/16B 可能是 2048/4096)
# 请根据 bird_adapter.py 中使用的模型调整此值
DEFAULT_TEXT_EMBED_DIM = 1536 

# --- RT Blocks (保持不变) ---
class RelationalTransformerBlock(nn.Module):
    """关系型 Transformer Block，包含四种注意力机制"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norms = nn.ModuleDict({
            "col": nn.LayerNorm(embed_dim),
            "feat": nn.LayerNorm(embed_dim),
            "nbr": nn.LayerNorm(embed_dim),
            "full": nn.LayerNorm(embed_dim),
            "ffn": nn.LayerNorm(embed_dim)
        })
        self.attns = nn.ModuleDict({
            "col": nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout),
            "feat": nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout),
            "nbr": nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout),
            "full": nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        })
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(), # 使用 SiLU 激活
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x, block_masks):
        """前向传播，应用四种注意力机制"""
        for l in ["col", "feat", "nbr", "full"]:
            residual = x
            norm_x = self.norms[l](x)
            mask_bool = block_masks[l]
            attn_mask = None
            if l != "full":
                attn_mask = ~mask_bool
            attn_out, _ = self.attns[l](
                query=norm_x, key=norm_x, value=norm_x,
                attn_mask=attn_mask, need_weights=False
            )
            x = residual + attn_out
        x = x + self.ffn(self.norms["ffn"](x))
        return x

class RTEmbedding(nn.Module):
    """关系型 Transformer 嵌入层，处理表、列、值的嵌入"""
    def __init__(self, channels, node_to_col_names, node_to_col_stats, table_list, text_embed_dim=DEFAULT_TEXT_EMBED_DIM):
        super().__init__()
        self.channels = channels
        self.node_to_col_names = node_to_col_names
        self.table_list = table_list
        self.table_to_idx = {t: i for i, t in enumerate(table_list)}
        
        self.table_emb = nn.Embedding(len(table_list), channels)
        self.col_embs = nn.ModuleDict()
        self.val_encoders = nn.ModuleDict()
        
        # 注意：这里不再自动加载 SentenceTransformer，而是假设 bird_adapter 已经传好了 embedding tensor
        # text_embed_dim 必须与 bird_adapter.py 中使用的模型输出维度一致
            
        for table in table_list:
            if table not in node_to_col_names: continue
                
            col_names = node_to_col_names[table]
            stats = node_to_col_stats.get(table, {})
            t_val_encs = nn.ModuleDict()
            t_col_embs = nn.ParameterDict()
            
            # 处理数值列
            for col in col_names.get(stype.numerical, []):
                t_val_encs[col] = nn.Sequential(nn.Linear(1, channels), nn.SiLU())
                t_col_embs[col] = nn.Parameter(torch.randn(channels))
            
            # 处理分类列
            for col in col_names.get(stype.categorical, []):
                # ... (保持原有的 categorical 统计逻辑) ...
                num_cats = 100 # 简化处理，实际应读取 stats
                if isinstance(stats, dict) and col in stats and stype.categorical in stats[col]:
                     if 'vocab' in stats[col][stype.categorical]:
                         num_cats = len(stats[col][stype.categorical]['vocab'])
                
                t_val_encs[col] = nn.Embedding(num_cats + 1, channels)
                t_col_embs[col] = nn.Parameter(torch.randn(channels))
            
            # 处理文本嵌入列 (这是 BIRD-SQL 的重点)
            for col in col_names.get(stype.text_embedded, []):
                # 关键修改：输入维度改为 text_embed_dim
                t_val_encs[col] = nn.Linear(text_embed_dim, channels) 
                t_col_embs[col] = nn.Parameter(torch.randn(channels))
                
            self.val_encoders[table] = t_val_encs
            self.col_embs[table] = t_col_embs

    def forward(self, tf_dict, edge_index_dict=None, table_to_node_offset=None):
        """前向传播，将 TensorFrame 转换为嵌入序列"""
        all_embs = []
        node_idxs_list = []
        table_idxs_list = []
        curr_node_offset = 0
        
        for table_name in self.table_list:
            if table_name not in tf_dict: continue
            tf = tf_dict[table_name]
            num_rows = tf.num_rows
            if num_rows == 0: continue
            
            t_idx = self.table_to_idx[table_name]
            t_emb = self.table_emb(torch.tensor(t_idx, device=tf.device))
            
            def add_token(val_enc, col_enc, val_data):
                token = val_enc(val_data) + col_enc + t_emb
                all_embs.append(token)
                node_idxs_list.append(torch.arange(curr_node_offset, curr_node_offset+num_rows, device=tf.device))
                table_idxs_list.append(torch.full((num_rows,), t_idx, device=tf.device))

            col_names = self.node_to_col_names[table_name]
            
            # 处理各类型列
            if stype.numerical in tf.feat_dict:
                feat = tf.feat_dict[stype.numerical]
                for i, col in enumerate(col_names.get(stype.numerical, [])):
                    add_token(self.val_encoders[table_name][col], self.col_embs[table_name][col], feat[:, i:i+1])
            
            if stype.categorical in tf.feat_dict:
                feat = tf.feat_dict[stype.categorical]
                for i, col in enumerate(col_names.get(stype.categorical, [])):
                    add_token(self.val_encoders[table_name][col], self.col_embs[table_name][col], feat[:, i])
            
            if stype.text_embedded in tf.feat_dict:
                feat = tf.feat_dict[stype.text_embedded]
                for i, col in enumerate(col_names.get(stype.text_embedded, [])):
                    # feat[:, i, :] 应该是 [Num_Rows, text_embed_dim]
                    add_token(self.val_encoders[table_name][col], self.col_embs[table_name][col], feat[:, i, :])
            
            curr_node_offset += num_rows
            
        if not all_embs: return None, None, None, None, None, 0
        
        x = torch.cat(all_embs, dim=0)
        node_idxs = torch.cat(node_idxs_list, dim=0)
        table_idxs = torch.cat(table_idxs_list, dim=0)
        
        # Col Idxs reconstruction
        col_idxs_tensor_list = []
        c_counter = 0
        for table_name in self.table_list:
            if table_name not in tf_dict: continue
            num = tf_dict[table_name].num_rows
            if num == 0: continue
            cnt = len(self.node_to_col_names[table_name].get(stype.numerical, [])) + \
                  len(self.node_to_col_names[table_name].get(stype.categorical, [])) + \
                  len(self.node_to_col_names[table_name].get(stype.text_embedded, []))
            for _ in range(cnt):
                col_idxs_tensor_list.append(torch.full((num,), c_counter, device=x.device))
                c_counter += 1
        col_idxs = torch.cat(col_idxs_tensor_list, dim=0)
        
        # 构建 f2p_nbr_idxs (简化版，复用之前逻辑)
        f2p_nbr_idxs = None
        if edge_index_dict and table_to_node_offset:
            max_parents = 16
            S = node_idxs.shape[0]
            device = x.device
            f2p_nbr_idxs = torch.full((S, max_parents), -1, dtype=torch.long, device=device)
            f2p_counts = torch.zeros(S, dtype=torch.long, device=device)
            
            node_to_cells = {}
            for cell_idx in range(S):
                node_idx = node_idxs[cell_idx].item()
                if node_idx not in node_to_cells: node_to_cells[node_idx] = []
                node_to_cells[node_idx].append(cell_idx)
            
            for edge_type, edge_index in edge_index_dict.items():
                src_table_name, rel_name, dst_table_name = edge_type
                if "rev_" in rel_name or "rev_fkey" in rel_name: continue
                if edge_index.shape[1] == 0: continue
                
                src_offset = table_to_node_offset.get(src_table_name, 0)
                dst_offset = table_to_node_offset.get(dst_table_name, 0)
                child_nodes_global = edge_index[0] + src_offset
                parent_nodes_global = edge_index[1] + dst_offset
                
                for i in range(edge_index.shape[1]):
                    child_node_global = child_nodes_global[i].item()
                    parent_node_global = parent_nodes_global[i].item()
                    if child_node_global not in node_to_cells: continue
                    for c_cell in node_to_cells[child_node_global]:
                        count = f2p_counts[c_cell].item()
                        if count < max_parents:
                            f2p_nbr_idxs[c_cell, count] = parent_node_global
                            f2p_counts[c_cell] += 1
        
        return x, node_idxs, col_idxs, table_idxs, f2p_nbr_idxs, curr_node_offset

# --- DeepSeek MoE Forward Fix (保持不变) ---
def deepseek_moe_forward_fixed(self, hidden_states):
    identity = hidden_states
    orig_shape = hidden_states.shape
    topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    flat_topk_idx = topk_idx.view(-1)
    if self.num_experts_per_tok > 1:
        hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
    y = torch.empty_like(hidden_states, dtype=hidden_states.dtype)
    for i, expert in enumerate(self.experts):
        idx_mask = (flat_topk_idx == i)
        if idx_mask.any():
            expert_out = expert(hidden_states[idx_mask])
            y[idx_mask] = expert_out.to(y.dtype)
    y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
    y = y.to(hidden_states.dtype).view(*orig_shape)
    y = y + identity
    return y

class DeepSeekRelationalModel(nn.Module):
    """DeepSeek 关系型 Transformer 主模型 (Generative / CoT Enabled)"""
    def __init__(self, data, col_stats_dict, args, task=None):
        super().__init__()
        self.task = task
        self.model_type = args.model_type
        self.main_device = torch.device("cuda:0") 
        self.hidden_dim = args.channels
        
        # 保存原始 data
        self.original_data = data
        
        col_names_dict = {}
        valid_table_list = []
        for node_type in data.node_types:
            if hasattr(data[node_type], 'tf') and data[node_type].tf is not None:
                col_names_dict[node_type] = data[node_type].tf.col_names_dict
                valid_table_list.append(node_type)
        
        filtered_col_stats_dict = {t: col_stats_dict.get(t, {}) for t in valid_table_list}
        
        # 获取 Embedding 维度，优先从 args 获取，否则用默认
        text_embed_dim = getattr(args, 'text_embed_dim', DEFAULT_TEXT_EMBED_DIM)

        # RT Tokenizer & Layers
        self.tokenizer = RTEmbedding(
            channels=self.hidden_dim,
            node_to_col_names=col_names_dict,
            node_to_col_stats=filtered_col_stats_dict,
            table_list=valid_table_list,
            text_embed_dim=text_embed_dim
        )
        self.rt_layers = nn.ModuleList([
            RelationalTransformerBlock(self.hidden_dim, num_heads=4, dropout=args.dropout)
            for _ in range(args.num_layers)
        ])

        # LLM Loading
        print(f"🚀 Loading LLM ({self.model_type}) for Generation...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_type, 
            device_map="auto",
            torch_dtype=torch.float16, 
            trust_remote_code=True
        )
        # 加载 Tokenizer (用于 generate 方法)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.model_type, trust_remote_code=True)
        self.llm_tokenizer.padding_side = 'left' # 这一步很关键
        if self.llm_tokenizer.pad_token is None:
             self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        # 修复 MoE Forward
        for module in self.llm.modules():
            if hasattr(module, 'gate') and hasattr(module, 'experts'):
                 module.forward = types.MethodType(deepseek_moe_forward_fixed, module)
        
        # 冻结 LLM 参数
        for param in self.llm.parameters(): param.requires_grad = False
            
        # Projector: RT Dim -> LLM Dim
        llm_dim = self.llm.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_dim, 1024),
            nn.SiLU(), 
            nn.Linear(1024, llm_dim)
        )
        
        # Attention-based Pooling for structure aggregation
        # 用于替代简单的 Mean Pooling，更好地聚合重要节点
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            batch_first=True,
            dropout=args.dropout
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        
        # 移除了 self.head (不再需要)

        # 移动到主设备
        self.tokenizer.to(self.main_device)
        self.rt_layers.to(self.main_device)
        self.projector.to(self.main_device)
        self.attention_pool.to(self.main_device)
        # pool_query 已经是 Parameter，直接移动到设备
        self.pool_query.data = self.pool_query.data.to(self.main_device)

    def create_masks(self, node_idxs, col_idxs, table_idxs, f2p_nbr_idxs=None, is_padding=None):
        """(保持原样) 构建四种注意力机制的mask"""
        # ... (Create Masks 代码与之前一致，省略以节省空间，请保留原有的实现) ...
        # 请务必保留原文件中的 create_masks 实现，不要删除！
        if node_idxs.dim() == 1:
            node_idxs = node_idxs.unsqueeze(0)
            col_idxs = col_idxs.unsqueeze(0)
            table_idxs = table_idxs.unsqueeze(0)
            if f2p_nbr_idxs is not None and f2p_nbr_idxs.dim() == 2:
                f2p_nbr_idxs = f2p_nbr_idxs.unsqueeze(0)
            if is_padding is not None and is_padding.dim() == 1:
                is_padding = is_padding.unsqueeze(0)
        B, S = node_idxs.shape
        if is_padding is not None:
            pad = (~is_padding[:, :, None]) & (~is_padding[:, None, :])
        else:
            pad = torch.ones((B, S, S), dtype=torch.bool, device=node_idxs.device)
        same_node = node_idxs[:, :, None] == node_idxs[:, None, :]
        same_col = col_idxs[:, :, None] == col_idxs[:, None, :]
        same_tab = table_idxs[:, :, None] == table_idxs[:, None, :]
        same_col_table = same_col & same_tab
        if f2p_nbr_idxs is not None:
            kv_in_f2p = (node_idxs[:, None, :, None] == f2p_nbr_idxs[:, :, None, :]).any(-1)
            q_in_f2p = (node_idxs[:, :, None, None] == f2p_nbr_idxs[:, None, :, :]).any(-1)
        else:
            kv_in_f2p = torch.zeros((B, S, S), dtype=torch.bool, device=node_idxs.device)
            q_in_f2p = torch.zeros((B, S, S), dtype=torch.bool, device=node_idxs.device)
        masks = {
            "feat": (same_node | kv_in_f2p) & pad,
            "nbr": q_in_f2p & pad,
            "col": same_col_table & pad,
            "full": pad
        }
        if B == 1:
            for key in masks: masks[key] = masks[key].squeeze(0)
        return masks

    def encode_structure(self, batch):
        """
        提取 RT 结构特征 (核心逻辑复用)
        Returns: 
            x_llm: [1, LLM_Dim] (Projected RT output) 或 None (如果出错)
        """
        try:
            # 1. 准备 TF Dict
            tf_dict = {}
            for node_type in batch.node_types:
                if hasattr(batch[node_type], 'tf') and batch[node_type].tf is not None:
                    tf_dict[node_type] = batch[node_type].tf
            
            # Fallback to original data if needed
            for node_type in batch.node_types:
                if node_type not in tf_dict and node_type in self.original_data.node_types:
                    original_tf = getattr(self.original_data[node_type], 'tf', None)
                    if original_tf is not None:
                        # 简化处理，实际应根据索引切片
                        tf_dict[node_type] = original_tf
                        
            if not tf_dict:
                print("⚠️ Warning: No TensorFrame found in batch")
                return None

            # 2. 计算 Offset
            table_to_node_offset = {}
            node_offset = 0
            for table_name in self.tokenizer.table_list:
                if table_name in tf_dict and tf_dict[table_name].num_rows > 0:
                    table_to_node_offset[table_name] = node_offset
                    node_offset += tf_dict[table_name].num_rows
            
            # 3. RT Tokenization
            tokenizer_output = self.tokenizer(
                tf_dict,
                edge_index_dict=batch.edge_index_dict,
                table_to_node_offset=table_to_node_offset
            )
            x, node_idxs, col_idxs, table_idxs, f2p_nbr_idxs, total_rows = tokenizer_output
            
            if x is None or x.shape[0] == 0:
                print("⚠️ Warning: Tokenizer returned empty output")
                return None
            
            # 4. RT Layers
            masks = self.create_masks(node_idxs, col_idxs, table_idxs, f2p_nbr_idxs)
            # 统一处理：确保有 batch 维度
            if x.dim() == 2:
                x = x.unsqueeze(0)  # [1, N, Dim]
            
            # 确保 masks 也有 batch 维度
            if isinstance(masks, dict):
                for key in masks:
                    if masks[key].dim() == 2:
                        masks[key] = masks[key].unsqueeze(0)
            
            for layer in self.rt_layers:
                x = layer(x, masks)
            
            # 移除 batch 维度（如果只有一个样本）
            if x.shape[0] == 1:
                x = x.squeeze(0)  # [N, Dim]
            
            # 5. Aggregate (Attention-based Pooling)
            # 使用 Attention Pooling 替代简单的 Mean Pooling
            # 这样可以自动学习哪些节点更重要
            if x.dim() == 2:
                x = x.unsqueeze(0)  # [1, N, Dim] for attention
            
            # 使用可学习的 query 进行 attention pooling
            query = self.pool_query.to(x.device).expand(x.shape[0], -1, -1)  # [1, 1, Dim]
            attn_out, attn_weights = self.attention_pool(
                query=query,
                key=x,
                value=x,
                need_weights=False
            )
            graph_emb = attn_out.squeeze(1)  # [1, RT_Dim]
            
            # 如果 attention 失败，fallback 到 mean pooling
            if graph_emb.shape[0] == 0:
                graph_emb = x.mean(dim=1)  # [1, RT_Dim]
            
            # 6. Project to LLM Space
            x_llm = self.projector(graph_emb)  # [1, LLM_Dim]
            
            return x_llm
            
        except Exception as e:
            print(f"❌ Error in encode_structure: {e}")
            import traceback
            traceback.print_exc()
            return None

    def forward(self, batch, input_ids=None, labels=None, **kwargs):
        """
        训练模式: Causal LM Loss
        Args:
            batch: HeteroData (Graph structure) 或 List[HeteroData] (批量处理)
            input_ids: [B, Seq_Len] (Question + SQL tokens)
            labels: [B, Seq_Len] (SQL tokens only, Question part masked as -100)
        Returns:
            loss: torch.Tensor 或 None (如果出错)
            logits: torch.Tensor 或 None
        """
        try:
            # 处理 batch 输入：支持单个 HeteroData 或列表
            if isinstance(batch, list):
                # 如果 batch 是列表，目前只处理第一个（未来可以扩展为真正的批量处理）
                if len(batch) == 0:
                    print("⚠️ Warning: Empty batch list")
                    return torch.tensor(0.0, device=self.main_device), None
                batch = batch[0]
            
            # 1. 计算结构特征 (Soft Prompt)
            if input_ids is None:
                print("⚠️ Warning: input_ids is None in forward")
                return torch.tensor(0.0, device=self.main_device), None
            
            bsz = input_ids.shape[0]
            
            rt_out = self.encode_structure(batch)  # [1, LLM_Dim]
            if rt_out is None:
                # 返回零损失而不是 None，避免训练中断
                print("⚠️ Warning: encode_structure returned None, returning zero loss")
                return torch.tensor(0.0, device=self.main_device, requires_grad=True), None
            
            # 使用 repeat 而不是 expand，确保内存安全
            structure_prompt = rt_out.repeat(bsz, 1)  # [B, LLM_Dim]
            structure_prompt = structure_prompt.unsqueeze(1)  # [B, 1, LLM_Dim]
            
            # 2. 准备 Text Embeddings
            # 这里的 input_ids 是 Question + SQL
            llm_embeds = self.llm.get_input_embeddings()(input_ids)  # [B, Seq, LLM_Dim]
            
            # 3. Concat: [Structure, Text]
            inputs_embeds = torch.cat([structure_prompt, llm_embeds], dim=1)  # [B, 1+Seq, LLM_Dim]
            
            # 4. 调整 Labels (Shift for Soft Prompt)
            if labels is not None:
                # Soft prompt 没有 label (-100)
                prefix_labels = torch.full((bsz, 1), -100, dtype=labels.dtype, device=labels.device)
                labels = torch.cat([prefix_labels, labels], dim=1)
            
            # 5. LLM Forward
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_hidden_states=False
            )
            
            return outputs.loss, outputs.logits
            
        except Exception as e:
            print(f"❌ Error in forward: {e}")
            import traceback
            traceback.print_exc()
            # 返回零损失，避免训练中断
            return torch.tensor(0.0, device=self.main_device, requires_grad=True), None

    @torch.no_grad()
    def generate(self, batch, question_text_list, max_new_tokens=512):
        """
        推理模式: CoT Generation
        Args:
            batch: HeteroData 或 List[HeteroData]
            question_text_list: List[str] 自然语言问题
        Returns:
            List[str]: 生成的文本列表
        """
        try:
            self.eval()
            
            # 处理 batch 输入
            if isinstance(batch, list):
                if len(batch) == 0:
                    return ["Error: Empty batch list"] * len(question_text_list)
                batch = batch[0]  # 目前只处理第一个，未来可扩展
            
            bsz = len(question_text_list)
            if bsz == 0:
                return []
            
            # 1. 结构特征
            rt_out = self.encode_structure(batch)
            if rt_out is None:
                return ["Error: No structure"] * bsz
            
            # 使用 repeat 确保内存安全
            structure_prompt = rt_out.repeat(bsz, 1).unsqueeze(1)  # [B, 1, LLM_Dim]
            
            # 2. 文本编码
            # 构造 Prompt: "Question: ... \n Answer:"
            # DeepSeek-R1 建议的 prompt 格式
            prompts = [f"Question: {q}\nAnswer:" for q in question_text_list]
            inputs = self.llm_tokenizer(prompts, return_tensors="pt", padding=True).to(self.main_device)
            
            text_embeds = self.llm.get_input_embeddings()(inputs.input_ids)
            
            # 3. Concat
            inputs_embeds = torch.cat([structure_prompt, text_embeds], dim=1)
            
            # Attention Mask (给 soft prompt 补 1)
            soft_prompt_mask = torch.ones((bsz, 1), device=self.main_device, dtype=inputs.attention_mask.dtype)
            attention_mask = torch.cat([soft_prompt_mask, inputs.attention_mask], dim=1)
            
            # 4. Generate
            # DeepSeek 会自动输出 <think>...</think>
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.llm_tokenizer.eos_token_id,
                do_sample=False  # Greedy decoding for stability in SQL
            )
            
            # 5. Decode
            decoded = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return decoded
            
        except Exception as e:
            print(f"❌ Error in generate: {e}")
            import traceback
            traceback.print_exc()
            return [f"Error: {str(e)}"] * len(question_text_list) if question_text_list else []