# BIRD-SQL 版本的 DeepSeek Relational Transformer

这个文件夹包含了从 My-DeepSeek-RT 迁移过来的核心代码，已移除 relbench 依赖，适配 BIRD-SQL 数据集。

## 📁 文件说明

### 核心模型文件
- **`model_clean.py`** - 核心模型文件
  - `RelationalTransformerBlock`: 四种注意力机制的 Transformer Block
  - `create_masks()`: 核心注意力 mask 生成函数
  - `RTEmbedding`: 表、列、值的嵌入层
  - `DeepSeekRelationalModel`: 主模型类（包含 LLM 集成）

### 辅助工具
- **`rt_utils.py`** - 图数据到序列的转换工具
- **`text_embedder.py`** - 文本编码器

### 工具类
- **`task_type.py`** - 自定义 TaskType 枚举（替换 relbench.base.TaskType）

## 🔄 主要变更

1. **移除 relbench 依赖**
   - 使用自定义 `TaskType` 枚举替代 `relbench.base.TaskType`
   - 移除了所有 `from relbench` 导入

2. **保留的核心功能**
   - 四种注意力机制（col, feat, nbr, full）
   - `create_masks()` 函数逻辑完全保留
   - `RelationalTransformerBlock` 完全保留
   - `RTEmbedding` 完全保留
   - 主模型结构完全保留

## 📝 使用说明

这个版本是为 BIRD-SQL 数据集准备的，需要：
1. 创建 BIRD-SQL 数据加载器（替换原来的 dataset.py）
2. 创建 BIRD-SQL 训练脚本（替换原来的 train_new.py）
3. 确保数据格式与模型输入格式匹配

## 🎯 下一步

- [ ] 创建 `bird_sql_dataset.py` - BIRD-SQL 数据加载器
- [ ] 创建 `train_bird_sql.py` - BIRD-SQL 训练脚本
- [ ] 创建 `bird_sql_utils.py` - BIRD-SQL 专用工具函数
