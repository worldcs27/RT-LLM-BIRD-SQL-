import os
import json
import sqlite3
import torch
import numpy as np
import networkx as nx
import hashlib
from functools import lru_cache
from typing import Dict, List, Tuple, Any, Optional
from torch_geometric.data import HeteroData
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from threading import Lock

# 尝试导入 torch_frame，如果没有则使用 Mock
try:
    from torch_frame import stype
except ImportError:
    class stype:
        numerical = "numerical"
        categorical = "categorical"
        text_embedded = "text_embedded"

# --- Mock TensorFrame 以兼容 model_clean.py ---
class MockTensorFrame:
    def __init__(self, feat_dict: Dict[str, torch.Tensor], col_names_dict: Dict[str, List[str]]):
        self.feat_dict = feat_dict
        self.col_names_dict = col_names_dict
        # 计算行数 (假设所有特征行数一致)
        self.num_rows = 0
        if feat_dict:
            first_key = next(iter(feat_dict))
            self.num_rows = feat_dict[first_key].shape[0]
        self.device = torch.device('cpu')

    def to(self, device):
        self.device = device
        for k, v in self.feat_dict.items():
            self.feat_dict[k] = v.to(device)
        return self

    def __getitem__(self, index):
        # 支持切片以兼容 RTEmbedding 的内部逻辑
        new_feat_dict = {}
        for k, v in self.feat_dict.items():
            new_feat_dict[k] = v[index]
        return MockTensorFrame(new_feat_dict, self.col_names_dict)

# --- 数据库连接池 ---
class DatabaseConnectionPool:
    """简单的数据库连接池，复用连接以减少开销"""
    def __init__(self, max_connections=10):
        self.max_connections = max_connections
        self.pools = {}  # {db_path: [conn1, conn2, ...]}
        self.lock = Lock()
    
    def get_connection(self, db_path: str) -> sqlite3.Connection:
        """获取数据库连接（从池中获取或创建新连接）"""
        with self.lock:
            if db_path not in self.pools:
                self.pools[db_path] = []
            
            pool = self.pools[db_path]
            if pool:
                conn = pool.pop()
                # 检查连接是否有效
                try:
                    conn.execute("SELECT 1")
                    return conn
                except:
                    # 连接已关闭，创建新的
                    pass
            
            # 创建新连接
            conn = sqlite3.connect(db_path, check_same_thread=False)
            return conn
    
    def return_connection(self, db_path: str, conn: sqlite3.Connection):
        """归还连接到池中"""
        with self.lock:
            if db_path not in self.pools:
                self.pools[db_path] = []
            
            pool = self.pools[db_path]
            if len(pool) < self.max_connections:
                pool.append(conn)
            else:
                conn.close()
    
    def close_all(self):
        """关闭所有连接"""
        with self.lock:
            for pool in self.pools.values():
                for conn in pool:
                    conn.close()
            self.pools.clear()

# --- 核心适配器类 ---
class BirdSQLAdapter:
    def __init__(self, bird_root_path: str, deepseek_model_path: str = "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"):
        """
        Args:
            bird_root_path: BIRD数据集根目录 (包含 train/train_tables.json 等)
            deepseek_model_path: 用于语义对齐的 DeepSeek 模型路径
        """
        self.root = bird_root_path
        self.train_tables_path = os.path.join(self.root, "train", "train_tables.json")
        self.train_db_root = os.path.join(self.root, "train", "train_databases")
        self.train_json_path = os.path.join(self.root, "train", "train.json")
        
        # 定义缓存目录
        self.cache_dir = os.path.join(self.root, "cache_deepseek_rt")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        print(f"🚀 Initializing BIRD-SQL Adapter...")
        print(f"   - Root: {self.root}")
        print(f"   - Cache Dir: {self.cache_dir}")
        print(f"   - Embedding Model: {deepseek_model_path}")

        # 1. 加载 DeepSeek 语义编码器
        # [内存优化] 使用 CPU offloading，减少 GPU 内存占用
        # 模型会加载到 CPU，只在需要时临时移动到 GPU
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                deepseek_model_path, 
                trust_remote_code=True,
                local_files_only=True
            )
            # [关键] 使用 CPU offloading，避免占用 GPU 内存
            self.model = AutoModel.from_pretrained(
                deepseek_model_path, 
                trust_remote_code=True, 
                device_map="cpu",  # 先加载到 CPU
                local_files_only=True,
                torch_dtype=torch.float16  # 使用 float16 减少内存
            )
            print("   ✅ Adapter model loaded to CPU (will move to GPU only when needed)")
        except Exception as e:
            print(f"⚠️ Warning: Failed to load local model: {e}")
            print("   Trying to load with network access...")
            self.tokenizer = AutoTokenizer.from_pretrained(deepseek_model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                deepseek_model_path, 
                trust_remote_code=True, 
                device_map="cpu",  # CPU offloading
                torch_dtype=torch.float16
            )
            
        self.model.eval()
        
        # 2. 解析 Schema Graph
        self.schemas, self.graphs = self._load_and_build_graphs()
        
        # 3. 预计算 Schema Embeddings (带磁盘缓存)
        self.schema_embeddings = self._precompute_schema_embeddings()
        
        # 4. 初始化数据库连接池
        self.db_pool = DatabaseConnectionPool(max_connections=10)
        
        # 5. 问题编码缓存
        self._question_cache = {}
        self._cache_lock = Lock()

    def _encode_text(self, texts: List[str], device="cuda", use_cache=True) -> torch.Tensor:
        """使用 DeepSeek 获取文本嵌入"""
        if len(texts) == 1 and use_cache:
            text = texts[0]
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            with self._cache_lock:
                if text_hash in self._question_cache:
                    return self._question_cache[text_hash]
        
        # [修复] 安全获取模型设备
        if self.model is None:
            raise RuntimeError("Model is not loaded. Cannot encode text.")
        
        # [内存优化] 如果模型在 CPU 上，临时移动到 GPU 进行编码
        model_was_on_cpu = False
        target_device = torch.device(device) if isinstance(device, str) else device
        
        try:
            first_param = next(self.model.parameters(), None)
            if first_param is not None:
                current_device = first_param.device
                # 如果模型在 CPU 上，且目标设备是 GPU，临时移动
                if current_device.type == 'cpu' and target_device.type == 'cuda':
                    model_was_on_cpu = True
                    # 临时移动到 GPU（只移动必要的层）
                    self.model = self.model.to(target_device)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  # 清理缓存
        except Exception as e:
            print(f"   ⚠️  Warning: Could not check/move model device: {e}")
        
        batch_size = 32
        all_embs = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
                # 将输入移动到目标设备
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
                last_hidden = outputs.last_hidden_state
                mask = inputs.attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * mask, 1)
                sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                embs = sum_embeddings / sum_mask
                all_embs.append(embs.cpu())
        
        result = torch.cat(all_embs, dim=0)
        
        # [内存优化] 如果模型是从 CPU 临时移动到 GPU 的，编码完成后移回 CPU
        if model_was_on_cpu:
            try:
                self.model = self.model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # 清理 GPU 缓存
            except Exception as e:
                print(f"   ⚠️  Warning: Could not move model back to CPU: {e}")
        
        if len(texts) == 1 and use_cache:
            text_hash = hashlib.md5(texts[0].encode('utf-8')).hexdigest()
            with self._cache_lock:
                if len(self._question_cache) >= 1000:
                    oldest_key = next(iter(self._question_cache))
                    del self._question_cache[oldest_key]
                self._question_cache[text_hash] = result
        
        return result

    def _load_and_build_graphs(self):
        """解析 JSON 构建 NetworkX 图"""
        print("📊 Parsing Schema and Building Graphs...")
        with open(self.train_tables_path, 'r', encoding='utf-8') as f:
            tables_data = json.load(f)
            
        schemas = {}
        graphs = {}
        
        for db in tables_data:
            db_id = db['db_id']
            schemas[db_id] = db
            G = nx.Graph() 
            for i, tbl_name in enumerate(db['table_names']):
                G.add_node(i, name=tbl_name, type='table')
            
            column_names = db['column_names']
            for src_col_idx, dst_col_idx in db['foreign_keys']:
                src_tbl_idx = column_names[src_col_idx][0]
                dst_tbl_idx = column_names[dst_col_idx][0]
                if src_tbl_idx != dst_tbl_idx:
                    G.add_edge(src_tbl_idx, dst_tbl_idx, type='fk')
            graphs[db_id] = G
            
        return schemas, graphs

    def _precompute_schema_embeddings(self):
        """预计算所有表名和列名的 Embedding (带磁盘缓存)"""
        cache_file = os.path.join(self.cache_dir, "schema_embeddings.pt")
        
        if os.path.exists(cache_file):
            print(f"💾 Loading cached schema embeddings from {cache_file}...")
            return torch.load(cache_file)
            
        print("🧠 Precomputing Schema Embeddings (First Time Run)...")
        cache = {}
        
        for db_id, schema in tqdm(self.schemas.items()):
            table_texts = [f"Table: {name}" for name in schema['table_names']]
            col_texts = []
            for idx, (tbl_idx, col_name) in enumerate(schema['column_names']):
                col_type = schema['column_types'][idx]
                col_texts.append(f"Column: {col_name} Type: {col_type}")
                
            t_embs = self._encode_text(table_texts)
            c_embs = self._encode_text(col_texts)
            
            cache[db_id] = {
                'table_embs': t_embs,
                'col_embs': c_embs
            }
        
        print(f"💾 Saving schema embeddings to {cache_file}...")
        torch.save(cache, cache_file)
        return cache

    def get_all_schema_metadata(self):
        """
        [新增] 获取所有数据库的 Schema 元数据
        用于模型初始化时构建参数 (RTEmbedding)，解决 'NoneType' 报错
        """
        meta_data = HeteroData()
        print("📦 Constructing Global Schema Metadata for Initialization...")
        
        # 使用 set 避免重复处理同名表
        processed_tables = set()
        
        for db_id, schema in self.schemas.items():
            table_names = schema['table_names']
            column_names = schema['column_names'] # [table_idx, col_name]
            
            for t_idx, t_name in enumerate(table_names):
                if t_name in processed_tables:
                    continue
                processed_tables.add(t_name)
                
                # 收集该表的所有列名
                curr_cols = []
                for c_idx, (tbl_idx, c_name) in enumerate(column_names):
                    if tbl_idx == t_idx:
                        curr_cols.append(c_name)
                
                # 构造 MockTensorFrame 元数据
                col_names_dict = {
                    stype.text_embedded: curr_cols
                }
                feat_dict = {} # 空字典，因为初始化不需要真实数据
                
                meta_data[t_name].tf = MockTensorFrame(feat_dict, col_names_dict)
                meta_data[t_name].num_nodes = 0
                
        return meta_data

    def prune_schema(self, question: str, db_id: str, top_k_tables=4) -> Tuple[List[int], List[int]]:
        """Question-Aware Schema Pruning (2-hop)"""
        if db_id not in self.schemas:
            return [], []
            
        schema = self.schemas[db_id]
        G = self.graphs[db_id]
        cached_embs = self.schema_embeddings[db_id]
        
        q_emb = self._encode_text([question], use_cache=True)[0]
        t_sim = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), cached_embs['table_embs'])
        
        num_tables = len(schema['table_names'])
        k = min(top_k_tables, num_tables)
        anchor_table_indices = torch.topk(t_sim, k).indices.tolist()
        
        active_tables = set(anchor_table_indices)
        neighbors = set()
        for t_idx in anchor_table_indices:
            if t_idx in G:
                for nbr in G.neighbors(t_idx):
                    neighbors.add(nbr)
        active_tables.update(neighbors)
        
        if len(active_tables) < 5:
            second_hop = set()
            for t_idx in neighbors:
                if t_idx in G:
                    for nbr in G.neighbors(t_idx):
                        second_hop.add(nbr)
            active_tables.update(second_hop)
            
        active_table_indices = sorted(list(active_tables))
        active_col_indices = []
        for col_idx, (tbl_idx, _) in enumerate(schema['column_names']):
            if tbl_idx in active_table_indices:
                active_col_indices.append(col_idx)
                
        return active_table_indices, active_col_indices
    
    def prune_schema_batch(self, questions: List[str], db_ids: List[str], top_k_tables=4) -> List[Tuple[List[int], List[int]]]:
        """批量 Schema Pruning"""
        if len(questions) != len(db_ids):
            raise ValueError("questions and db_ids must have the same length")
        
        results = []
        unique_questions = list(set(questions))
        question_to_emb = {}
        if unique_questions:
            batch_embs = self._encode_text(unique_questions, use_cache=False)
            for q, emb in zip(unique_questions, batch_embs):
                question_to_emb[q] = emb
        
        for question, db_id in zip(questions, db_ids):
            if db_id not in self.schemas:
                results.append(([], []))
                continue
            schema = self.schemas[db_id]
            G = self.graphs[db_id]
            cached_embs = self.schema_embeddings[db_id]
            q_emb = question_to_emb[question]
            
            t_sim = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), cached_embs['table_embs'])
            num_tables = len(schema['table_names'])
            k = min(top_k_tables, num_tables)
            anchor_table_indices = torch.topk(t_sim, k).indices.tolist()
            
            active_tables = set(anchor_table_indices)
            neighbors = set()
            for t_idx in anchor_table_indices:
                if t_idx in G:
                    for nbr in G.neighbors(t_idx):
                        neighbors.add(nbr)
            active_tables.update(neighbors)
            
            if len(active_tables) < 5:
                second_hop = set()
                for t_idx in neighbors:
                    if t_idx in G:
                        for nbr in G.neighbors(t_idx):
                            second_hop.add(nbr)
                active_tables.update(second_hop)
            
            active_table_indices = sorted(list(active_tables))
            active_col_indices = [col_idx for col_idx, (tbl_idx, _) in enumerate(schema['column_names']) if tbl_idx in active_table_indices]
            results.append((active_table_indices, active_col_indices))
        return results

    def get_sample_hetero_data(self, question: str, db_id: str):
        """构建单个 HeteroData 对象"""
        active_table_idxs, active_col_idxs = self.prune_schema(question, db_id)
        return self._build_hetero_data_single(db_id, active_table_idxs, active_col_idxs)
    
    def get_sample_hetero_data_batch(self, questions: List[str], db_ids: List[str]) -> List[HeteroData]:
        """批量构建 HeteroData 对象"""
        pruning_results = self.prune_schema_batch(questions, db_ids)
        results = []
        for question, db_id, (active_table_idxs, active_col_idxs) in zip(questions, db_ids, pruning_results):
            try:
                data = self._build_hetero_data_single(db_id, active_table_idxs, active_col_idxs)
                results.append(data)
            except Exception as e:
                print(f"⚠️ Warning: Failed to build HeteroData for question '{question[:50]}...' in {db_id}: {e}")
                results.append(HeteroData()) 
        return results
    
    def _build_hetero_data_single(self, db_id: str, active_table_idxs: List[int], active_col_idxs: List[int]) -> HeteroData:
        """为单个问题构建 HeteroData (内部方法)"""
        schema = self.schemas[db_id]
        data = HeteroData()
        col_names = schema['column_names']
        
        for t_idx in active_table_idxs:
            table_name = schema['table_names'][t_idx]
            curr_table_col_idxs = [i for i in active_col_idxs if col_names[i][0] == t_idx]
            
            if not curr_table_col_idxs:
                continue
            
            db_path = os.path.join(self.train_db_root, db_id, f"{db_id}.sqlite")
            conn = self.db_pool.get_connection(db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 3")
                rows = cursor.fetchall()
            except Exception as e:
                # print(f"⚠️ Warning: Failed to read from table {table_name}: {e}")
                rows = []
            finally:
                self.db_pool.return_connection(db_path, conn)
            
            num_rows = max(len(rows), 1)
            
            table_col_embs = []
            for c_idx in curr_table_col_idxs:
                emb = self.schema_embeddings[db_id]['col_embs'][c_idx]
                table_col_embs.append(emb)
            
            if not table_col_embs:
                continue
            
            col_feats = torch.stack(table_col_embs)
            col_feats = col_feats.unsqueeze(0).expand(num_rows, -1, -1)
            
            feat_dict = {stype.text_embedded: col_feats.float()}
            c_names = [col_names[i][1] for i in curr_table_col_idxs]
            col_names_dict = {stype.text_embedded: c_names}
            
            data[table_name].tf = MockTensorFrame(feat_dict, col_names_dict)
            data[table_name].num_nodes = num_rows
        
        for src_col, dst_col in schema['foreign_keys']:
            src_t_idx = col_names[src_col][0]
            dst_t_idx = col_names[dst_col][0]
            
            if src_t_idx in active_table_idxs and dst_t_idx in active_table_idxs:
                src_name = schema['table_names'][src_t_idx]
                dst_name = schema['table_names'][dst_t_idx]
                
                if src_name in data.node_types and dst_name in data.node_types:
                    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                    data[src_name, "fkey", dst_name].edge_index = edge_index
                    data[dst_name, "rev_fkey", src_name].edge_index = edge_index
        
        return data
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'db_pool'):
            self.db_pool.close_all()

# --- 测试代码 ---
if __name__ == "__main__":
    BIRD_ROOT = "/data/cuishuai/datasets/text-to-sql/BIRD-SQL" 
    TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 
    if os.path.exists(BIRD_ROOT):
        adapter = BirdSQLAdapter(BIRD_ROOT, TEST_MODEL)
        q = "How many customers?"
        db_id = list(adapter.schemas.keys())[0]
        t_idxs, c_idxs = adapter.prune_schema(q, db_id)
        print(f"Selected Tables: {t_idxs}")