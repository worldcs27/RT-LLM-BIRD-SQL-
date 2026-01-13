#!/bin/bash

# ================= 配置区域 =================
# ✅ 使用4张空闲的GPU (2, 4, 6, 7) 分散模型负载
# 这些GPU相对空闲（约1.6 GB占用），可以分散模型到多张卡上
export CUDA_VISIBLE_DEVICES=2,4,6,7

# [内存优化] 减少内存碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 项目路径配置
BIRD_ROOT="/data/cuishuai/datasets/text-to-sql/BIRD-SQL"
OUTPUT_DIR="./checkpoints_deepseek_rt_4bit" # 改名区分

# ... (其余配置保持不变) ...
MODEL_PATH="deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
TEXT_EMBED_DIM=2048
RT_CHANNELS=1024
RT_LAYERS=8
EPOCHS=10
LR=5e-5
GRAD_ACC_STEPS=32 

# ================= 启动命令 =================
echo "🚀 Starting DeepSeek-RT Training on 4 GPUs (2,4,6,7) with 4-bit Quantization..."
echo "   📊 Model will be distributed across multiple GPUs to reduce memory pressure"
# ... (其余保持不变) ...
python -u train_bird_sql.py \
    --bird_root "$BIRD_ROOT" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_PATH" \
    --channels $RT_CHANNELS \
    --num_layers $RT_LAYERS \
    --dropout 0.1 \
    --text_embed_dim $TEXT_EMBED_DIM \
    --epochs $EPOCHS \
    --lr $LR \
    --grad_acc_steps $GRAD_ACC_STEPS \
    --save_limit 3 \
    2>&1 | tee train_log.txt

echo "🎉 Training finished! Logs saved to train_log.txt"