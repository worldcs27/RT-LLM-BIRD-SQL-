import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import random

# 导入我们构建的核心模块
from bird_adapter import BirdSQLAdapter
from model import DeepSeekRelationalModel

# 设置随机种子以保证复现性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class BirdTrainDataset(Dataset):
    """BIRD-SQL 训练数据集"""
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def prepare_inputs(batch_data, adapter, tokenizer, device, max_length=1024):
    """
    数据预处理核心函数
    """
    item = batch_data[0]
    question = item['question']
    sql = item['SQL']
    db_id = item['db_id']
    
    # --- 1. 获取图数据 (RT Input) ---
    try:
        hetero_data = adapter.get_sample_hetero_data(question, db_id)
        hetero_data_list = [hetero_data]
    except Exception as e:
        print(f"⚠️ Error building graph for {db_id}: {e}")
        return None, None, None

    # --- 2. 构建文本输入 (LLM Input) ---
    prompt = f"Question: {question}\nAnswer:"
    target = f"{sql}<|endoftext|>"
    full_text = prompt + " " + target
    
    encoded = tokenizer(
        full_text, 
        max_length=max_length, 
        truncation=True, 
        return_tensors="pt",
        padding=False 
    )
    input_ids = encoded.input_ids.to(device) # [1, Seq]
    
    # --- 3. 构建 Labels ---
    labels = input_ids.clone()
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    prompt_len = prompt_ids.shape[1]
    
    if prompt_len < labels.shape[1]:
        labels[:, :prompt_len] = -100
    else:
        return None, None, None

    return hetero_data_list, input_ids, labels

def main():
    parser = argparse.ArgumentParser(description="Train DeepSeek-RT on BIRD-SQL")
    
    # 路径配置
    parser.add_argument("--bird_root", type=str, default="/data/cuishuai/datasets/text-to-sql/BIRD-SQL", help="BIRD数据集根目录")
    parser.add_argument("--model_path", type=str, default="deepseek-ai/DeepSeek-Coder-V2-Lite-Base", help="DeepSeek模型路径")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="模型保存路径")
    
    # 模型配置
    parser.add_argument("--model_type", type=str, default="deepseek-ai/DeepSeek-Coder-V2-Lite-Base")
    parser.add_argument("--channels", type=int, default=512, help="RT隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=4, help="RT层数")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--text_embed_dim", type=int, default=2048, help="文本嵌入维度")
    
    # 训练配置
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--grad_acc_steps", type=int, default=16, help="梯度累积步数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_limit", type=int, default=3, help="最多保留几个checkpoint")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training Device: {device}")

    # --- 1. 初始化 Adapter ---
    print("📦 Initializing Adapter...")
    adapter = BirdSQLAdapter(
        bird_root_path=args.bird_root,
        deepseek_model_path=args.model_path
    )
    
    # --- 2. 加载数据 ---
    print("📚 Loading Dataset...")
    train_dataset = BirdTrainDataset(os.path.join(args.bird_root, "train", "train.json"))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)
    
    # --- 3. 初始化模型 ---
    print("🧠 Initializing DeepSeek-RT Model...")
    
    # [修复关键] 获取全量 Schema 元数据，用于初始化 Embedding 层
    # 这样模型就知道 BIRD 里一共有多少张表，每张表有哪些列
    print("🔍 Fetching global schema metadata for initialization...")
    init_data = adapter.get_all_schema_metadata()
    
    # [内存优化] 在加载主模型前，清理 GPU 缓存
    # 注意：adapter 模型已加载到 CPU，不会占用 GPU 内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"   💾 GPU Memory Before Loading Main Model: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB / {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    col_stats_dict = {} 
    
    model = DeepSeekRelationalModel(
        data=init_data,  # [修复关键] 传入全量元数据，不再是 None
        col_stats_dict=col_stats_dict, 
        args=args
    )
    
    # --- 4. 优化器与调度器 ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"💪 Trainable Parameters: {len(trainable_params)}")
    
    optimizer = AdamW(trainable_params, lr=args.lr)
    total_steps = len(train_loader) * args.epochs // args.grad_acc_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    
    llm_tokenizer = model.llm_tokenizer
    
    # --- 5. 训练循环 ---
    print("🔥 Starting Training...")
    global_step = 0
    tr_loss = 0.0
    optimizer.zero_grad()
    model.train()
    
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        epoch_iterator = tqdm(train_loader, desc="Training")
        
        for step, batch_data in enumerate(epoch_iterator):
            # 1. 准备输入
            hetero_data_list, input_ids, labels = prepare_inputs(batch_data, adapter, llm_tokenizer, model.main_device)
            
            if hetero_data_list is None:
                continue 
                
            # 2. Forward
            loss, logits = model(
                batch=hetero_data_list,
                input_ids=input_ids,
                labels=labels
            )
            
            # 3. 梯度累积
            if args.grad_acc_steps > 1:
                loss = loss / args.grad_acc_steps
            
            loss.backward()
            tr_loss += loss.item()
            
            # 4. Optimizer Step
            if (step + 1) % args.grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if global_step % args.log_interval == 0:
                    avg_loss = tr_loss / args.log_interval
                    epoch_iterator.set_postfix(loss=avg_loss, step=global_step)
                    tr_loss = 0.0
        
        # Epoch End Saving
        save_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(save_path, exist_ok=True)
        print(f"💾 Saving checkpoint to {save_path}...")
        
        torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
        with open(os.path.join(save_path, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    print("🎉 Training Completed!")

if __name__ == "__main__":
    main()