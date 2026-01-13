# 离线模式使用指南

## 问题描述

即使模型文件已经提前下载到本地缓存，`transformers` 库在加载模型时仍然可能尝试连接网络，导致 `ReadTimeoutError`。

## 原因分析

1. **文件验证**: `transformers` 库在加载模型时会尝试验证文件完整性
2. **版本检查**: 可能尝试检查模型版本或更新信息
3. **索引文件**: 某些索引文件可能触发网络请求
4. **设备映射**: 使用 `device_map="auto"` 时可能尝试在线获取设备信息

## 解决方案

### 方案 1: 使用 `local_files_only=True` (已实现)

代码已经更新，在 `bird_adapter.py` 和 `model.py` 中添加了 `local_files_only=True` 参数：

```python
# bird_adapter.py
self.tokenizer = AutoTokenizer.from_pretrained(
    deepseek_model_path, 
    trust_remote_code=True,
    local_files_only=True  # 强制使用本地文件
)

# model.py
self.llm = AutoModelForCausalLM.from_pretrained(
    self.model_type, 
    device_map="auto",
    torch_dtype=torch.float16, 
    trust_remote_code=True,
    local_files_only=True  # 强制使用本地文件
)
```

如果本地文件加载失败，代码会自动回退到允许网络连接的模式。

### 方案 2: 设置环境变量 (可选)

如果需要全局强制离线模式，可以设置环境变量：

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

### 方案 3: 使用本地路径 (推荐)

如果仍然遇到问题，可以直接使用本地路径：

```python
# 使用本地路径而不是模型名称
local_model_path = "/data/cuishuai/hf_cache/hub/deepseek-ai/DeepSeek-Coder-V2-Lite-Base"

adapter = BirdSQLAdapter(
    bird_root_path=BIRD_ROOT,
    deepseek_model_path=local_model_path  # 使用本地路径
)
```

## 验证模型文件完整性

确保以下文件存在：

```bash
# 检查模型文件
ls -lh /data/cuishuai/hf_cache/hub/deepseek-ai/DeepSeek-Coder-V2-Lite-Base/*.safetensors
# 应该有 4 个文件

# 检查配置文件
ls -lh /data/cuishuai/hf_cache/hub/deepseek-ai/DeepSeek-Coder-V2-Lite-Base/*.json
# 应该有 6 个文件（包括 config.json, tokenizer.json 等）
```

## 常见问题

### Q: 为什么即使设置了 `local_files_only=True` 仍然尝试连接网络？

A: 可能是某些依赖文件缺失，或者 transformers 库版本较旧。可以：
1. 检查所有必需文件是否存在
2. 更新 transformers 库到最新版本
3. 使用本地路径而不是模型名称

### Q: 如何确认模型文件完整？

A: 运行以下命令检查：

```bash
# 检查文件数量
find /data/cuishuai/hf_cache/hub/deepseek-ai/DeepSeek-Coder-V2-Lite-Base -type f | wc -l

# 检查总大小（应该约 30GB）
du -sh /data/cuishuai/hf_cache/hub/deepseek-ai/DeepSeek-Coder-V2-Lite-Base
```

### Q: 如果仍然遇到网络连接问题怎么办？

A: 可以尝试：
1. 使用本地路径而不是模型名称
2. 设置环境变量 `TRANSFORMERS_OFFLINE=1`
3. 检查网络代理设置
4. 增加超时时间（在代码中设置）

## 当前配置

- **模型路径**: `deepseek-ai/DeepSeek-Coder-V2-Lite-Base`
- **本地缓存**: `/data/cuishuai/hf_cache/hub/deepseek-ai/DeepSeek-Coder-V2-Lite-Base`
- **离线模式**: 已启用 (`local_files_only=True`)
- **回退机制**: 如果离线加载失败，自动尝试允许网络连接
