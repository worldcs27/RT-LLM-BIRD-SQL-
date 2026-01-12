"""
自定义 TaskType 枚举，用于替换 relbench.base.TaskType
"""
from enum import Enum

class TaskType(Enum):
    """任务类型枚举"""
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    
    # 兼容 relbench 的命名
    @property
    def value(self):
        return self._value_
