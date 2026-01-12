"""
BIRD-SQL 版本的 DeepSeek Relational Transformer 模型
移除了 relbench 依赖，使用自定义 TaskType
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from torch_frame import stype
from task_type import TaskType  # 使用自定义 TaskType
import types

SENTENCE_TRANSFORMER_PATH = "sentence-transformers/all-MiniLM-L12-v2"

# --- RT Blocks ---
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
            nn.SiLU(),
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
    def __init__(self, channels, node_to_col_names, node_to_col_stats, table_list):
        super().__init__()
        self.channels = channels
        self.node_to_col_names = node_to_col_names
        self.table_list = table_list
        self.table_to_idx = {t: i for i, t in enumerate(table_list)}
        
        self.table_emb = nn.Embedding(len(table_list), channels)
        self.col_embs = nn.ModuleDict()
        self.val_encoders = nn.ModuleDict()
        
        print(f"📦 Loading Text Encoder: {SENTENCE_TRANSFORMER_PATH}...")
        self.text_model = AutoModel.from_pretrained(SENTENCE_TRANSFORMER_PATH)
        for p in self.text_model.parameters(): p.requires_grad = False
            
        for table in table_list:
            # 确保表存在于 node_to_col_names 中
            if table not in node_to_col_names:
                print(f"⚠️ 警告: 表 '{table}' 不在 node_to_col_names 中，跳过")
                continue
            if table not in node_to_col_stats:
                print(f"⚠️ 警告: 表 '{table}' 不在 node_to_col_stats 中，跳过")
                continue
                
            col_names = node_to_col_names[table]
            stats = node_to_col_stats.get(table, {})
            t_val_encs = nn.ModuleDict()
            t_col_embs = nn.ParameterDict()  # 使用 ParameterDict 存储 Parameter
            
            # 处理数值列
            for col in col_names.get(stype.numerical, []):
                t_val_encs[col] = nn.Sequential(nn.Linear(1, channels), nn.SiLU())
                t_col_embs[col] = nn.Parameter(torch.randn(channels))
            
            # 处理分类列
            for col in col_names.get(stype.categorical, []):
                try:
                    if isinstance(stats, dict) and col in stats:
                        if isinstance(stats[col], dict) and stype.categorical in stats[col]:
                            if isinstance(stats[col][stype.categorical], dict) and 'vocab' in stats[col][stype.categorical]:
                                num_cats = len(stats[col][stype.categorical]['vocab'])
                            else:
                                num_cats = 100
                        else:
                            num_cats = 100
                    else:
                        num_cats = 100
                except (KeyError, TypeError, AttributeError) as e:
                    print(f"⚠️ 警告: 列 '{col}' 的统计信息访问出错 ({e})，使用默认值 100")
                    num_cats = 100
                t_val_encs[col] = nn.Embedding(num_cats + 1, channels)
                t_col_embs[col] = nn.Parameter(torch.randn(channels))
            
            # 处理文本嵌入列
            for col in col_names.get(stype.text_embedded, []):
                t_val_encs[col] = nn.Linear(300, channels) 
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
        
        # 构建 f2p_nbr_idxs (每个单元格的父节点列表)
        f2p_nbr_idxs = None
        if edge_index_dict and table_to_node_offset:
            max_parents = 16
            S = node_idxs.shape[0]
            device = x.device
            f2p_nbr_idxs = torch.full((S, max_parents), -1, dtype=torch.long, device=device)
            f2p_counts = torch.zeros(S, dtype=torch.long, device=device)
            
            # 构建节点到单元格的映射
            node_to_cells = {}
            for cell_idx in range(S):
                node_idx = node_idxs[cell_idx].item()
                if node_idx not in node_to_cells:
                    node_to_cells[node_idx] = []
                node_to_cells[node_idx].append(cell_idx)
            
            # 从 edge_index_dict 构建 f2p 连接
            for edge_type, edge_index in edge_index_dict.items():
                src_table_name, rel_name, dst_table_name = edge_type
                
                # 只处理 f2p 边，跳过反向边
                if "rev_" in rel_name or "rev_fkey" in rel_name:
                    continue
                
                if edge_index.shape[1] == 0:
                    continue
                
                src_offset = table_to_node_offset.get(src_table_name, 0)
                dst_offset = table_to_node_offset.get(dst_table_name, 0)
                
                child_nodes_local = edge_index[0]
                parent_nodes_local = edge_index[1]
                
                child_nodes_global = child_nodes_local + src_offset
                parent_nodes_global = parent_nodes_local + dst_offset
                
                # 对于每个 (child, parent) 对，更新 child 的所有单元格的 f2p_nbr_idxs
                for i in range(edge_index.shape[1]):
                    child_node_global = child_nodes_global[i].item()
                    parent_node_global = parent_nodes_global[i].item()
                    
                    if child_node_global not in node_to_cells:
                        continue
                    
                    child_cells = node_to_cells[child_node_global]
                    for c_cell in child_cells:
                        count = f2p_counts[c_cell].item()
                        if count < max_parents:
                            f2p_nbr_idxs[c_cell, count] = parent_node_global
                            f2p_counts[c_cell] += 1
        
        return x, node_idxs, col_idxs, table_idxs, f2p_nbr_idxs, curr_node_offset

# --- DeepSeek MoE Forward Fix ---
def deepseek_moe_forward_fixed(self, hidden_states):
    """修复 DeepSeek MoE 的 forward 方法"""
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
    """DeepSeek 关系型 Transformer 主模型"""
    def __init__(self, data, col_stats_dict, args, task):
        super().__init__()
        self.task = task
        self.model_type = args.model_type
        self.main_device = torch.device("cuda:0") 
        self.hidden_dim = args.channels
        
        # 保存原始 data 以便从 batch 中恢复 tf
        self.original_data = data
        
        col_names_dict = {}
        valid_table_list = []
        for node_type in data.node_types:
            if hasattr(data[node_type], 'tf') and data[node_type].tf is not None:
                col_names_dict[node_type] = data[node_type].tf.col_names_dict
                valid_table_list.append(node_type)
        
        # 确保 col_stats_dict 也只包含有效的表
        filtered_col_stats_dict = {}
        for table_name in valid_table_list:
            if table_name in col_stats_dict:
                filtered_col_stats_dict[table_name] = col_stats_dict[table_name]
            else:
                print(f"⚠️ 警告: 表 '{table_name}' 不在 col_stats_dict 中，将使用空统计信息")
                filtered_col_stats_dict[table_name] = {}
        
        # RT Tokenizer & Layers
        self.tokenizer = RTEmbedding(
            channels=self.hidden_dim,
            node_to_col_names=col_names_dict,
            node_to_col_stats=filtered_col_stats_dict,
            table_list=valid_table_list
        )
        self.rt_layers = nn.ModuleList([
            RelationalTransformerBlock(self.hidden_dim, num_heads=4, dropout=args.dropout)
            for _ in range(args.num_layers)
        ])

        # LLM (自动多卡切分)
        print(f"🚀 Loading LLM with device_map='auto' across GPUs...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_type, 
            device_map="auto",
            torch_dtype=torch.float16, 
            trust_remote_code=True
        )
        for module in self.llm.modules():
            if hasattr(module, 'gate') and hasattr(module, 'experts'):
                 module.forward = types.MethodType(deepseek_moe_forward_fixed, module)
        for param in self.llm.parameters(): param.requires_grad = False
            
        # Projector & Head
        llm_dim = self.llm.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_dim, 1024),
            nn.SiLU(), 
            nn.Linear(1024, llm_dim)
        )
        
        out_dim = 1
        if hasattr(task, 'num_labels'): out_dim = task.num_labels
        self.head = nn.Linear(llm_dim, out_dim)

        # 移动到主设备
        self.tokenizer.to(self.main_device)
        self.rt_layers.to(self.main_device)
        self.projector.to(self.main_device)
        self.head.to(self.main_device)

    def create_masks(self, node_idxs, col_idxs, table_idxs, f2p_nbr_idxs=None, is_padding=None):
        """
        构建四种注意力机制的mask（核心函数）：
        - col: 同一列同一表的单元格
        - feat: 同一节点内的单元格 OR 通过外键连接的父节点单元格
        - nbr: 通过外键连接的子节点单元格（邻居）
        - full: 所有单元格（全连接）
        
        Args:
            node_idxs: [S] 或 [B, S] 每个单元格属于哪个节点（全局索引）
            col_idxs: [S] 或 [B, S] 每个单元格属于哪一列
            table_idxs: [S] 或 [B, S] 每个单元格属于哪张表
            f2p_nbr_idxs: [S, Max_Parents] 或 [B, S, Max_Parents] 每个单元格的父节点列表，-1表示无效
            is_padding: [S] 或 [B, S] padding mask，True表示padding
        """
        # 处理单batch情况（添加batch维度）
        if node_idxs.dim() == 1:
            node_idxs = node_idxs.unsqueeze(0)
            col_idxs = col_idxs.unsqueeze(0)
            table_idxs = table_idxs.unsqueeze(0)
            if f2p_nbr_idxs is not None and f2p_nbr_idxs.dim() == 2:
                f2p_nbr_idxs = f2p_nbr_idxs.unsqueeze(0)
            if is_padding is not None and is_padding.dim() == 1:
                is_padding = is_padding.unsqueeze(0)
        
        B, S = node_idxs.shape
        
        # Padding mask
        if is_padding is not None:
            pad = (~is_padding[:, :, None]) & (~is_padding[:, None, :])
        else:
            pad = torch.ones((B, S, S), dtype=torch.bool, device=node_idxs.device)
        
        # 同一节点内的单元格
        same_node = node_idxs[:, :, None] == node_idxs[:, None, :]
        
        # 同一列且同一表
        same_col = col_idxs[:, :, None] == col_idxs[:, None, :]
        same_tab = table_idxs[:, :, None] == table_idxs[:, None, :]
        same_col_table = same_col & same_tab
        
        # f2p 连接（使用 f2p_nbr_idxs）
        if f2p_nbr_idxs is not None:
            kv_in_f2p = (node_idxs[:, None, :, None] == f2p_nbr_idxs[:, :, None, :]).any(-1)
            q_in_f2p = (node_idxs[:, :, None, None] == f2p_nbr_idxs[:, None, :, :]).any(-1)
        else:
            kv_in_f2p = torch.zeros((B, S, S), dtype=torch.bool, device=node_idxs.device)
            q_in_f2p = torch.zeros((B, S, S), dtype=torch.bool, device=node_idxs.device)
        
        # 构建四种mask
        masks = {
            "feat": (same_node | kv_in_f2p) & pad,
            "nbr": q_in_f2p & pad,
            "col": same_col_table & pad,
            "full": pad
        }
        
        # 如果是单batch，移除batch维度
        if B == 1:
            for key in masks:
                masks[key] = masks[key].squeeze(0)
        
        return masks

    def forward(self, batch, entity_table, **kwargs):
        """前向传播"""
        # 从 batch 或原始 data 中构建 tf_dict
        tf_dict = {}
        
        # 首先尝试从 batch 中获取 tf
        for node_type in batch.node_types:
            if hasattr(batch[node_type], 'tf') and batch[node_type].tf is not None:
                tf_dict[node_type] = batch[node_type].tf
        
        # 如果 batch 中没有 tf，从原始 data 中获取
        for node_type in batch.node_types:
            if node_type not in tf_dict:
                if node_type in self.original_data.node_types:
                    original_tf = getattr(self.original_data[node_type], 'tf', None)
                    if original_tf is not None:
                        original_tf = self.original_data[node_type].tf
                        if hasattr(batch[node_type], 'n_id'):
                            original_indices = batch[node_type].n_id
                            try:
                                tf_dict[node_type] = original_tf[original_indices]
                            except Exception as e:
                                print(f"⚠️ 警告: 无法从原始 tf 中索引节点 {node_type}: {e}")
                                continue
                        else:
                            num_nodes = batch[node_type].num_nodes
                            if num_nodes > 0 and num_nodes <= original_tf.num_rows:
                                try:
                                    tf_dict[node_type] = original_tf[:num_nodes]
                                except Exception as e:
                                    print(f"⚠️ 警告: 无法从原始 tf 中切片节点 {node_type}: {e}")
                                    continue
        
        if not tf_dict:
            print(f"⚠️ 警告: 无法从 batch 或原始 data 中获取 tf_dict")
            return None
        
        # 计算每个表在batch中的节点偏移量
        table_to_node_offset = {}
        node_offset = 0
        for table_name in self.tokenizer.table_list:
            if table_name in tf_dict and tf_dict[table_name].num_rows > 0:
                table_to_node_offset[table_name] = node_offset
                node_offset += tf_dict[table_name].num_rows
        
        # 传递 edge_index_dict 和 table_to_node_offset 以构建 f2p_nbr_idxs
        x, node_idxs, col_idxs, table_idxs, f2p_nbr_idxs, total_rows = self.tokenizer(
            tf_dict,
            edge_index_dict=batch.edge_index_dict,
            table_to_node_offset=table_to_node_offset
        )
        if x is None: return None
        
        # 构建mask
        masks = self.create_masks(node_idxs, col_idxs, table_idxs, f2p_nbr_idxs)
        x = x.unsqueeze(0)
        for layer in self.rt_layers:
            x = layer(x, masks)
        x = x.squeeze(0)
        
        # Aggregate
        out_nodes = torch.zeros(total_rows, self.hidden_dim, device=x.device)
        out_nodes.index_reduce_(0, node_idxs, x, reduce="mean", include_self=False)
        target_start = 0
        for t in list(batch.node_types):
            if t == entity_table: break
            target_start += batch[t].num_nodes
        target_indices = torch.arange(target_start, target_start + batch[entity_table].batch_size, device=x.device)
        target_emb = out_nodes[target_indices]
        
        # Project & Bridge to LLM
        x_llm = self.projector(target_emb)
        
        bsz = x_llm.shape[0]
        if not hasattr(self, 'bos_token_id'):
            self.bos_token_id = self.llm.config.bos_token_id if self.llm.config.bos_token_id else 1
        
        first_param = next(self.llm.parameters())
        llm_device = first_param.device
        
        x_llm = x_llm.to(llm_device)
        bos_idx = torch.tensor([[self.bos_token_id]], device=llm_device)
        bos_emb = self.llm.get_input_embeddings()(bos_idx).expand(bsz, -1, -1)
        
        inputs_embeds = torch.cat([bos_emb, x_llm.unsqueeze(1)], dim=1)
        
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            outputs = self.llm(
                inputs_embeds=inputs_embeds, 
                output_hidden_states=True,
                use_cache=False
            )
            
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        last_hidden = last_hidden.to(self.main_device)
        logits = self.head(last_hidden)
        
        return logits

    def compute_loss(self, pred, target):
        """计算损失"""
        target = target.to(pred.device)
        # 确保 target 是正确类型
        task_type = getattr(self.task, 'task_type', None)
        if task_type is None:
            # 默认使用回归
            task_type = TaskType.REGRESSION
        elif isinstance(task_type, str):
            # 兼容字符串形式的 task_type
            if task_type == "regression" or task_type == TaskType.REGRESSION.value:
                task_type = TaskType.REGRESSION
            elif task_type == "binary_classification" or task_type == TaskType.BINARY_CLASSIFICATION.value:
                task_type = TaskType.BINARY_CLASSIFICATION
            elif task_type == "multiclass_classification":
                task_type = TaskType.MULTICLASS_CLASSIFICATION
            else:
                task_type = TaskType.REGRESSION
        
        if task_type == TaskType.REGRESSION:
            target = target.float()
            return F.huber_loss(pred.squeeze(), target.squeeze())
        else:
            num_labels = getattr(self.task, 'num_labels', 1)
            if num_labels == 1:
                target = target.float()
                return F.binary_cross_entropy_with_logits(pred.squeeze(), target.squeeze())
            else:
                target = target.long()
                return F.cross_entropy(pred, target)
