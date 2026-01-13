"""
BIRD-SQL 版本的 DeepSeek Relational Transformer (支持 CoT 生成)
- 移除了 RelBench 依赖
- 移除了分类头，改为 Causal LM 生成头
- 增加了 generate() 方法支持思维链 (CoT)
- 修复了 'type' 列名冲突问题
- [本次修复] 使用 device_map={"": 0} 强制单卡加载，解决 auto 策略误判显存不足的问题
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from task_type import TaskType
import types

# 尝试导入 torch_frame，如果没有则使用 Mock
try:
    from torch_frame import stype
except ImportError:
    class stype:
        numerical = "numerical"
        categorical = "categorical"
        text_embedded = "text_embedded"

# 默认使用的文本编码维度
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
    def __init__(self, channels, node_to_col_names, node_to_col_stats, table_list, text_embed_dim=DEFAULT_TEXT_EMBED_DIM):
        super().__init__()
        self.channels = channels
        self.node_to_col_names = node_to_col_names
        self.table_list = table_list
        self.table_to_idx = {t: i for i, t in enumerate(table_list)}
        
        self.table_emb = nn.Embedding(len(table_list), channels)
        self.col_embs = nn.ModuleDict()
        self.val_encoders = nn.ModuleDict()
        
        for table in table_list:
            if table not in node_to_col_names: continue
                
            col_names = node_to_col_names[table]
            stats = node_to_col_stats.get(table, {})
            t_val_encs = nn.ModuleDict()
            t_col_embs = nn.ParameterDict()
            
            # 处理数值列
            for col in col_names.get(stype.numerical, []):
                safe_col = self._safe_key(col)
                t_val_encs[safe_col] = nn.Sequential(nn.Linear(1, channels), nn.SiLU())
                t_col_embs[safe_col] = nn.Parameter(torch.randn(channels))
            
            # 处理分类列
            for col in col_names.get(stype.categorical, []):
                safe_col = self._safe_key(col)
                num_cats = 100 
                if isinstance(stats, dict) and col in stats and stype.categorical in stats[col]:
                     if 'vocab' in stats[col][stype.categorical]:
                         num_cats = len(stats[col][stype.categorical]['vocab'])
                
                t_val_encs[safe_col] = nn.Embedding(num_cats + 1, channels)
                t_col_embs[safe_col] = nn.Parameter(torch.randn(channels))
            
            # 处理文本嵌入列
            for col in col_names.get(stype.text_embedded, []):
                safe_col = self._safe_key(col)
                t_val_encs[safe_col] = nn.Linear(text_embed_dim, channels) 
                t_col_embs[safe_col] = nn.Parameter(torch.randn(channels))
                
            self.val_encoders[table] = t_val_encs
            self.col_embs[table] = t_col_embs

    def _safe_key(self, key):
        """防止列名与 PyTorch 内置方法冲突"""
        safe_key = str(key).replace('.', '_').replace('/', '_').replace('\\', '_')
        safe_key = safe_key.replace(' ', '_').replace('-', '_')
        return f"c_{safe_key}"

    def forward(self, tf_dict, edge_index_dict=None, table_to_node_offset=None):
        """前向传播"""
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
                    safe_col = self._safe_key(col)
                    add_token(self.val_encoders[table_name][safe_col], self.col_embs[table_name][safe_col], feat[:, i:i+1])
            
            if stype.categorical in tf.feat_dict:
                feat = tf.feat_dict[stype.categorical]
                for i, col in enumerate(col_names.get(stype.categorical, [])):
                    safe_col = self._safe_key(col)
                    add_token(self.val_encoders[table_name][safe_col], self.col_embs[table_name][safe_col], feat[:, i])
            
            if stype.text_embedded in tf.feat_dict:
                feat = tf.feat_dict[stype.text_embedded]
                for i, col in enumerate(col_names.get(stype.text_embedded, [])):
                    safe_col = self._safe_key(col)
                    add_token(self.val_encoders[table_name][safe_col], self.col_embs[table_name][safe_col], feat[:, i, :])
            
            curr_node_offset += num_rows
            
        if not all_embs: return None, None, None, None, None, 0
        
        x = torch.cat(all_embs, dim=0)
        node_idxs = torch.cat(node_idxs_list, dim=0)
        table_idxs = torch.cat(table_idxs_list, dim=0)
        
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
        
        # [多GPU支持] 检测可用的GPU数量
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            # 如果指定了CUDA_VISIBLE_DEVICES，使用所有可见的GPU
            visible_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if visible_gpus:
                # 解析可见的GPU列表（如 "2,4,6,7"）
                gpu_list = [int(x.strip()) for x in visible_gpus.split(',') if x.strip()]
                num_gpus = len(gpu_list) if gpu_list else num_gpus
            self.num_gpus = min(num_gpus, 4)  # 最多使用4张GPU
            self.main_device = torch.device("cuda:0")  # 主设备仍然是cuda:0（相对于CUDA_VISIBLE_DEVICES）
            print(f"   🎯 Using {self.num_gpus} GPU(s) for model distribution")
        else:
            self.num_gpus = 0
            self.main_device = torch.device("cpu")
        
        self.hidden_dim = args.channels
        
        # 保存原始 data
        self.original_data = data
        
        col_names_dict = {}
        valid_table_list = []
        if data is not None:
            for node_type in data.node_types:
                if hasattr(data[node_type], 'tf') and data[node_type].tf is not None:
                    col_names_dict[node_type] = data[node_type].tf.col_names_dict
                    valid_table_list.append(node_type)
        
        filtered_col_stats_dict = {t: col_stats_dict.get(t, {}) for t in valid_table_list}
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

        # LLM Loading (4-bit Quantization)
        print(f"🚀 Loading LLM ({self.model_type}) with 4-bit Quantization (Forced on GPU 0)...")
        
        # [内存优化] 在加载前清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"   💾 GPU Memory Before Loading: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB / {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
        # [配置] 4-bit 量化
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # [多GPU支持] 将LLM分散到多个GPU
        # [关键修复] 使用CPU offloading + 严格的max_memory限制
        if self.num_gpus > 1:
            # 使用CPU offloading，让部分层在CPU上，减少GPU内存压力
            # GPU 0: 限制为3GB（为RT组件预留更多空间）
            # GPU 1-3: 各4GB
            # CPU: 剩余层
            max_memory = {
                0: "3GB",  # GPU 0限制更严格
                1: "4GB",
                2: "4GB", 
                3: "4GB"
            }
            # 添加CPU offloading
            max_memory["cpu"] = "30GB"  # CPU可以存储更多
            
            device_map = "auto"  # 让accelerate自动分配，但受max_memory限制
            print(f"   🎯 Distributing LLM with CPU offloading across {self.num_gpus} GPUs")
            print(f"      GPU 0: max 3GB (reserved for RT components)")
            print(f"      GPU 1-3: max 4GB each")
            print(f"      CPU: overflow layers")
        else:
            device_map = {"": 0}
            max_memory = {0: "22GB"} if torch.cuda.is_available() else None

        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_type, 
            device_map=device_map,
            quantization_config=bnb_config,
            torch_dtype=torch.float16, 
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
            max_memory=max_memory if torch.cuda.is_available() else None
        )
        
        # [内存优化] 加载后再次清理缓存，并检查所有GPU的内存使用
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"   ✅ GPU Memory After Loading:")
            for i in range(self.num_gpus):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"      GPU {i}: {allocated:.2f} GB / {reserved:.2f} GB")
            
            # [关键修复] 如果GPU 0内存占用过高，强制清理并尝试释放
            gpu0_reserved = torch.cuda.memory_reserved(0) / 1024**3
            if gpu0_reserved > 10:  # 如果GPU 0预留超过10GB
                print(f"   ⚠️  GPU 0 has high memory reservation ({gpu0_reserved:.2f} GB), attempting to free...")
                # 强制同步所有CUDA操作
                torch.cuda.synchronize()
                # 多次清理缓存
                for _ in range(3):
                    torch.cuda.empty_cache()
                # 检查清理后的内存
                gpu0_reserved_after = torch.cuda.memory_reserved(0) / 1024**3
                print(f"   ✅ After cleanup: GPU 0 reserved = {gpu0_reserved_after:.2f} GB")
        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            self.model_type, 
            trust_remote_code=True,
            local_files_only=True
        )
        self.llm_tokenizer.padding_side = 'left'
        if self.llm_tokenizer.pad_token is None:
             self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        # 修复 MoE Forward
        for module in self.llm.modules():
            if hasattr(module, 'gate') and hasattr(module, 'experts'):
                 module.forward = types.MethodType(deepseek_moe_forward_fixed, module)
        
        # 冻结 LLM 参数
            
        llm_dim = self.llm.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_dim, 1024),
            nn.SiLU(), 
            nn.Linear(1024, llm_dim)
        )
        
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            batch_first=True,
            dropout=args.dropout
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        
        # [多GPU支持] 将RT组件分散到多个GPU
        # [关键修复] 如果所有GPU内存都不足，将部分组件放在CPU上
        if torch.cuda.is_available() and self.num_gpus > 1:
            print(f"   📦 Distributing RT components across {self.num_gpus} GPUs...")
            
            # 检查所有GPU的可用内存
            gpu_available = []
            for i in range(self.num_gpus):
                gpu_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                gpu_reserved = torch.cuda.memory_reserved(i) / 1024**3
                gpu_available.append(gpu_total - gpu_reserved)
                print(f"   💾 GPU {i}: {gpu_reserved:.2f} GB reserved / {gpu_total:.2f} GB total ({gpu_available[i]:.2f} GB available)")
            
            # 找到可用内存最多的GPU
            best_gpu = max(range(self.num_gpus), key=lambda i: gpu_available[i])
            print(f"   🎯 Best GPU for RT components: GPU {best_gpu} ({gpu_available[best_gpu]:.2f} GB available)")
            
            # 如果所有GPU可用内存都很少，使用CPU offloading
            if all(avail < 1.0 for avail in gpu_available):
                print(f"   ⚠️  All GPUs have limited memory, using CPU for some RT components")
                tokenizer_device = torch.device("cpu")
                rt_layers_device = torch.device("cpu")
                projector_device = torch.device("cpu")
                attention_device = torch.device("cpu")
            else:
                # 将组件分散到可用内存最多的GPU
                tokenizer_device = torch.device(f"cuda:{best_gpu}")
                # rt_layers分散到GPU 1和2（如果可用）
                rt_layers_device = torch.device("cuda:1") if self.num_gpus > 1 and gpu_available[1] > 1.0 else torch.device("cuda:0")
                projector_device = torch.device("cuda:2") if self.num_gpus > 2 and gpu_available[2] > 1.0 else torch.device("cuda:0")
                attention_device = torch.device("cuda:3") if self.num_gpus > 3 and gpu_available[3] > 1.0 else torch.device("cuda:0")
            
            # 1. 较小的组件（pool_query放在主设备）
            try:
                self.pool_query.data = self.pool_query.data.to(self.main_device)
                torch.cuda.empty_cache()
            except RuntimeError:
                # 如果失败，放在CPU
                self.pool_query.data = self.pool_query.data.to(torch.device("cpu"))
            
            # 2. Tokenizer
            try:
                self.tokenizer.to(tokenizer_device)
                torch.cuda.empty_cache()
                print(f"   ✅ Tokenizer moved to {tokenizer_device}")
            except RuntimeError as e:
                print(f"   ❌ Failed to move tokenizer: {e}, keeping on CPU")
                self.tokenizer.to(torch.device("cpu"))
            
            # 3. RT Layers（分散到不同GPU或CPU）
            mid_layer = len(self.rt_layers) // 2
            try:
                for i in range(mid_layer):
                    self.rt_layers[i] = self.rt_layers[i].to(rt_layers_device)
                torch.cuda.empty_cache()
                print(f"   ✅ RT layers (first half) moved to {rt_layers_device}")
            except RuntimeError as e:
                print(f"   ⚠️  Failed to move RT layers to {rt_layers_device}: {e}, using CPU")
                for i in range(mid_layer):
                    self.rt_layers[i] = self.rt_layers[i].to(torch.device("cpu"))
            
            try:
                for i in range(mid_layer, len(self.rt_layers)):
                    self.rt_layers[i] = self.rt_layers[i].to(rt_layers_device)
                torch.cuda.empty_cache()
            except RuntimeError:
                for i in range(mid_layer, len(self.rt_layers)):
                    self.rt_layers[i] = self.rt_layers[i].to(torch.device("cpu"))
            
            # 4. Projector
            try:
                self.projector.to(projector_device)
                torch.cuda.empty_cache()
                print(f"   ✅ Projector moved to {projector_device}")
            except RuntimeError:
                self.projector.to(torch.device("cpu"))
                print(f"   ⚠️  Projector moved to CPU")
            
            # 5. Attention Pool
            try:
                self.attention_pool.to(attention_device)
                torch.cuda.empty_cache()
                print(f"   ✅ Attention pool moved to {attention_device}")
            except RuntimeError:
                self.attention_pool.to(torch.device("cpu"))
                print(f"   ⚠️  Attention pool moved to CPU")
            
            # 打印各GPU内存使用
            for i in range(self.num_gpus):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"   ✅ GPU {i}: {allocated:.2f} GB / {reserved:.2f} GB")
        elif torch.cuda.is_available():
            # 单GPU模式：分批移动组件
            print("   📦 Moving components to GPU (with memory optimization)...")
            self.pool_query.data = self.pool_query.data.to(self.main_device)
            torch.cuda.empty_cache()
            self.projector.to(self.main_device)
            torch.cuda.empty_cache()
            self.attention_pool.to(self.main_device)
            torch.cuda.empty_cache()
            self.rt_layers.to(self.main_device)
            torch.cuda.empty_cache()
            self.tokenizer.to(self.main_device)
            torch.cuda.empty_cache()
            print(f"   ✅ All components moved. Final GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB / {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        else:
            # CPU 模式，直接移动
            self.tokenizer.to(self.main_device)
            self.rt_layers.to(self.main_device)
            self.projector.to(self.main_device)
            self.attention_pool.to(self.main_device)
            self.pool_query.data = self.pool_query.data.to(self.main_device)

    def create_masks(self, node_idxs, col_idxs, table_idxs, f2p_nbr_idxs=None, is_padding=None):
        """(保持原样)"""
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
        """提取 RT 结构特征"""
        try:
            tf_dict = {}
            for node_type in batch.node_types:
                if hasattr(batch[node_type], 'tf') and batch[node_type].tf is not None:
                    tf_dict[node_type] = batch[node_type].tf
            
            for node_type in batch.node_types:
                if node_type not in tf_dict and node_type in self.original_data.node_types:
                    original_tf = getattr(self.original_data[node_type], 'tf', None)
                    if original_tf is not None:
                        tf_dict[node_type] = original_tf
                        
            if not tf_dict: return None

            table_to_node_offset = {}
            node_offset = 0
            for table_name in self.tokenizer.table_list:
                if table_name in tf_dict and tf_dict[table_name].num_rows > 0:
                    table_to_node_offset[table_name] = node_offset
                    node_offset += tf_dict[table_name].num_rows
            
            tokenizer_output = self.tokenizer(
                tf_dict,
                edge_index_dict=batch.edge_index_dict,
                table_to_node_offset=table_to_node_offset
            )
            x, node_idxs, col_idxs, table_idxs, f2p_nbr_idxs, total_rows = tokenizer_output
            
            if x is None or x.shape[0] == 0: return None
            
            masks = self.create_masks(node_idxs, col_idxs, table_idxs, f2p_nbr_idxs)
            if x.dim() == 2: x = x.unsqueeze(0)
            if isinstance(masks, dict):
                for key in masks:
                    if masks[key].dim() == 2: masks[key] = masks[key].unsqueeze(0)
            
            # [多GPU支持] 确保数据在正确的设备上，并处理分散的RT layers
            for i, layer in enumerate(self.rt_layers):
                # 获取当前layer所在的设备
                layer_device = next(layer.parameters()).device
                # 将数据移动到layer的设备
                if x.device != layer_device:
                    x = x.to(layer_device)
                    masks = {k: v.to(layer_device) for k, v in masks.items()}
                x = layer(x, masks)
            
            if x.shape[0] == 1: x = x.squeeze(0)
            
            if x.dim() == 2: x = x.unsqueeze(0)
            
            # [多GPU支持] 确保attention_pool在正确的设备上
            attention_device = next(self.attention_pool.parameters()).device
            if x.device != attention_device:
                x = x.to(attention_device)
            query = self.pool_query.to(attention_device).expand(x.shape[0], -1, -1)
            attn_out, attn_weights = self.attention_pool(query=query, key=x, value=x, need_weights=False)
            graph_emb = attn_out.squeeze(1)
            
            if graph_emb.shape[0] == 0: graph_emb = x.mean(dim=1)
            
            # [多GPU支持] 确保projector在正确的设备上
            projector_device = next(self.projector.parameters()).device
            if graph_emb.device != projector_device:
                graph_emb = graph_emb.to(projector_device)
            x_llm = self.projector(graph_emb)
            return x_llm
            
        except Exception as e:
            print(f"❌ Error in encode_structure: {e}")
            return None

    def forward(self, batch, input_ids=None, labels=None, **kwargs):
        """训练模式: Causal LM Loss"""
        try:
            if isinstance(batch, list):
                if len(batch) == 0: return torch.tensor(0.0, device=self.main_device, requires_grad=True), None
                batch = batch[0]
            
            if input_ids is None: return torch.tensor(0.0, device=self.main_device, requires_grad=True), None
            
            bsz = input_ids.shape[0]
            
            rt_out = self.encode_structure(batch)
            if rt_out is None:
                return torch.tensor(0.0, device=self.main_device, requires_grad=True), None
            
            structure_prompt = rt_out.repeat(bsz, 1).unsqueeze(1)
            llm_embeds = self.llm.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([structure_prompt, llm_embeds], dim=1)
            
            if labels is not None:
                prefix_labels = torch.full((bsz, 1), -100, dtype=labels.dtype, device=labels.device)
                labels = torch.cat([prefix_labels, labels], dim=1)
            
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_hidden_states=False
            )
            return outputs.loss, outputs.logits
            
        except Exception as e:
            print(f"❌ Error in forward: {e}")
            return torch.tensor(0.0, device=self.main_device, requires_grad=True), None

    @torch.no_grad()
    def generate(self, batch, question_text_list, max_new_tokens=512):
        """推理模式: CoT Generation"""
        try:
            self.eval()
            if isinstance(batch, list):
                if len(batch) == 0: return ["Error: Empty batch list"] * len(question_text_list)
                batch = batch[0]
            
            bsz = len(question_text_list)
            if bsz == 0: return []
            
            rt_out = self.encode_structure(batch)
            if rt_out is None: return ["Error: No structure"] * bsz
            
            structure_prompt = rt_out.repeat(bsz, 1).unsqueeze(1)
            prompts = [f"Question: {q}\nAnswer:" for q in question_text_list]
            inputs = self.llm_tokenizer(prompts, return_tensors="pt", padding=True).to(self.main_device)
            text_embeds = self.llm.get_input_embeddings()(inputs.input_ids)
            
            inputs_embeds = torch.cat([structure_prompt, text_embeds], dim=1)
            soft_prompt_mask = torch.ones((bsz, 1), device=self.main_device, dtype=inputs.attention_mask.dtype)
            attention_mask = torch.cat([soft_prompt_mask, inputs.attention_mask], dim=1)
            
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.llm_tokenizer.eos_token_id,
                do_sample=False 
            )
            decoded = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return decoded
        except Exception as e:
            print(f"❌ Error in generate: {e}")
            return [f"Error: {str(e)}"] * len(question_text_list) if question_text_list else []