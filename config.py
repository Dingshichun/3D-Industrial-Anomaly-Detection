import os
from dataclasses import dataclass
from typing import List

@dataclass
class DatasetConfig:
    """数据集配置参数，用于指定数据来源及基本预处理尺寸"""
    raw_data_root: str = "./data/MVTec3D-AD"  # 原始 MVTec 3D-AD 数据集根目录
    categories: List[str] = None              # 需要检测的缺陷物品类别列表
    image_size: int = 256                     # 统一调整后的点云深度图和RGB图像大小
    train_batch_size: int = 4
    val_batch_size: int = 4
    test_batch_size: int = 4
    num_workers: int = 4
    
    def __post_init__(self):
        # 默认使用所有 5 个类别进行评估
        if self.categories is None:
            self.categories = ['dowel', 'cable_gland', 'tire', 'rope', 'foam']

@dataclass
class ModelConfig:
    """模型配置参数"""
    backbone: str = 'resnet18'                # PatchCore 与 AST 特征提取的统一骨干网络
    f_coreset: float = 0.01                   # 核心集下采样比例(未使用，改由 Evaluator 控制)

@dataclass
class TrainingConfig:
    """训练循环配置参数（如果在 AST 中需自定义 epochs 等可扩展于此）"""
    pass
