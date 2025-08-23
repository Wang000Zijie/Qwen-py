# strategy_library.py

import os
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from sklearn.metrics.pairwise import cosine_distances # 需要安装 scikit-learn: pip install scikit-learn
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingState:
    """
    表征当前训练状态的特征向量
    """
    epoch: int
    avg_train_loss: float
    grad_norm: float
    progress: float  # 训练进度 (completed_steps / total_steps)
    val_loss: Optional[float] = None
    val_metric: Optional[float] = None # 例如 accuracy, f1 等

@dataclass
class Strategy:
    """
    存储超参数配置和策略
    """
    # 优化器相关
    learning_rate: float
    optimizer_name: str # e.g., 'adamw_torch', 'adamw_8bit'
    
    # 训练相关
    gradient_accumulation_steps: int
    scheduler_type: str # e.g., 'cosine', 'cosine_with_restarts', 'polynomial'
    scheduler_kwargs: Dict[str, Any] # 额外的调度器参数
    
    # 其他可能的策略
    # batch_size: int # 注意：动态改变 batch_size 较复杂，这里暂不实现
    # data_augmentation: str # 数据增强策略

@dataclass
class StrategyRecord:
    """
    一条完整的策略记录，包含状态、策略和结果
    """
    state: TrainingState
    strategy: Strategy
    outcome: Dict[str, Any] # 例如: {'final_val_loss': 1.0, 'convergence_speed': 'fast', 'stability': 'high', 'notes': '...'}

class StrategyLibrary:
    """
    检索式超参数优化的核心库
    """
    def __init__(self, library_path: str = "strategy_library.json"):
        """
        初始化策略库
        :param library_path: 策略库文件的路径
        """
        self.library_path = library_path
        self.records: List[StrategyRecord] = []
        self.load_library()
        logger.info(f"Strategy Library initialized with {len(self.records)} records.")

    def _state_to_vector(self, state: TrainingState) -> np.ndarray:
        """
        将训练状态转换为可用于计算相似度的数值向量
        """
        # 简单的向量化方法，可以根据需要调整权重或标准化
        vector = np.array([
            state.epoch,
            state.avg_train_loss,
            state.grad_norm,
            state.progress,
            state.val_loss if state.val_loss is not None else 0.0, # 用0填充缺失值
            state.val_metric if state.val_metric is not None else 0.0
        ])
        return vector

    def add_record(self, state: TrainingState, strategy: Strategy, outcome: Dict[str, Any]):
        """
        添加一条新的训练记录到库中
        """
        record = StrategyRecord(state=state, strategy=strategy, outcome=outcome)
        self.records.append(record)
        logger.info(f"Added new record for state (epoch={state.epoch}, loss={state.avg_train_loss:.4f})")
        self.save_library() # 每次添加都保存，确保不丢数据

    def retrieve_best_strategy(self, current_state: TrainingState, k: int = 3) -> Optional[Strategy]:
        """
        根据当前状态检索最佳策略
        :param current_state: 当前训练状态
        :param k: 考虑最近的 k 个相似记录
        :return: 推荐的策略，如果库为空则返回 None
        """
        if not self.records:
            logger.warning("Strategy library is empty. Cannot retrieve strategy.")
            return None

        current_vector = self._state_to_vector(current_state)
        
        # 计算与所有历史记录的距离
        distances = []
        valid_records = [] # 只考虑状态完整的记录
        for record in self.records:
            # 简单过滤：确保关键字段存在
            if record.state.avg_train_loss is not None and record.state.grad_norm is not None:
                record_vector = self._state_to_vector(record.state)
                # 使用余弦距离
                distance = cosine_distances([current_vector], [record_vector])[0][0]
                distances.append(distance)
                valid_records.append(record)
        
        if not valid_records:
            logger.warning("No valid records found in library for comparison.")
            return None

        # 找到 k 个最近邻
        distances = np.array(distances)
        k = min(k, len(distances))
        nearest_indices = np.argpartition(distances, k-1)[:k]
        
        # 简单的策略推荐：选择 outcome 中 val_loss 最小的（假设越小越好）
        # 或者可以根据其他指标（如 stability, convergence_speed）进行加权
        best_record = None
        best_outcome_metric = float('inf')
        
        logger.info(f"Found {k} nearest neighbors for current state:")
        for i in nearest_indices:
            record = valid_records[i]
            distance = distances[i]
            # 假设我们关注验证集 loss
            val_loss = record.outcome.get('final_val_loss', float('inf'))
            logger.info(f"  Neighbor (dist={distance:.4f}): val_loss={val_loss:.4f}, lr={record.strategy.learning_rate}, accum={record.strategy.gradient_accumulation_steps}")
            
            if val_loss < best_outcome_metric:
                best_outcome_metric = val_loss
                best_record = record
        
        if best_record:
            logger.info(f"Recommended strategy based on neighbor with val_loss={best_outcome_metric:.4f}")
            return best_record.strategy
        else:
            logger.warning("Could not determine best strategy from neighbors.")
            return None

    def save_library(self):
        """
        将策略库保存到文件
        """
        try:
            # 将 dataclass 转换为字典以便序列化
            serializable_records = []
            for record in self.records:
                record_dict = {
                    'state': asdict(record.state),
                    'strategy': asdict(record.strategy),
                    'outcome': record.outcome
                }
                serializable_records.append(record_dict)
            
            with open(self.library_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_records, f, indent=4, ensure_ascii=False)
            logger.info(f"Strategy library saved to {self.library_path}")
        except Exception as e:
            logger.error(f"Failed to save strategy library: {e}")

    def load_library(self):
        """
        从文件加载策略库
        """
        if os.path.exists(self.library_path):
            try:
                with open(self.library_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.records = []
                for item in data:
                    # 从字典重建 dataclass 对象
                    state = TrainingState(**item['state'])
                    strategy = Strategy(**item['strategy'])
                    outcome = item['outcome']
                    record = StrategyRecord(state=state, strategy=strategy, outcome=outcome)
                    self.records.append(record)
                logger.info(f"Strategy library loaded from {self.library_path} with {len(self.records)} records.")
            except Exception as e:
                logger.error(f"Failed to load strategy library: {e}. Starting with an empty library.")
                self.records = []
        else:
            logger.info(f"Strategy library file {self.library_path} not found. Starting with an empty library.")

# --- 辅助函数，用于初始化一些基本策略 ---

def create_initial_strategies() -> List[Tuple[TrainingState, Strategy, Dict[str, Any]]]:
    """
    创建一些初始的策略记录，用于库的初始化
    这些是基于经验和你提供的训练方法.txt中的推荐组合
    """
    initial_records = []

    # 记录1: 稳定提升组合 (来自训练方法.txt)
    state1 = TrainingState(epoch=1, avg_train_loss=2.0, grad_norm=1.5, progress=0.05, val_loss=2.1)
    strategy1 = Strategy(
        learning_rate=1e-4,
        optimizer_name="adamw_8bit",
        gradient_accumulation_steps=4,
        scheduler_type="cosine_with_restarts",
        scheduler_kwargs={"num_cycles": 3}
    )
    outcome1 = {"final_val_loss": 1.2, "convergence_speed": "medium", "stability": "high", "notes": "Stable baseline from training_methods.txt"}
    initial_records.append((state1, strategy1, outcome1))

    # 记录2: 激进提升组合 (来自训练方法.txt)
    state2 = TrainingState(epoch=5, avg_train_loss=1.0, grad_norm=0.8, progress=0.3, val_loss=1.1)
    strategy2 = Strategy(
        learning_rate=5e-5,
        optimizer_name="lion", # 假设已安装
        gradient_accumulation_steps=2,
        scheduler_type="polynomial",
        scheduler_kwargs={"power": 1.0, "lr_end": 1e-6}
    )
    outcome2 = {"final_val_loss": 1.0, "convergence_speed": "fast", "stability": "medium", "notes": "Aggressive combo, faster convergence"}
    initial_records.append((state2, strategy2, outcome2))

    # 记录3: 内存优化组合 (来自训练方法.txt)
    state3 = TrainingState(epoch=10, avg_train_loss=0.8, grad_norm=0.5, progress=0.7, val_loss=0.9)
    strategy3 = Strategy(
        learning_rate=3e-5,
        optimizer_name="adamw_8bit",
        gradient_accumulation_steps=8, # 更大的累积步数以节省内存
        scheduler_type="linear", # 简单线性衰减
        scheduler_kwargs={}
    )
    outcome3 = {"final_val_loss": 0.95, "convergence_speed": "slow", "stability": "high", "notes": "Memory efficient, slower but stable"}
    initial_records.append((state3, strategy3, outcome3))

    return initial_records

# --- 示例用法 ---
if __name__ == '__main__':
    # 创建库实例
    lib = StrategyLibrary("example_strategy_library.json")
    
    # 如果库是空的，添加一些初始策略
    if not lib.records:
        initial_strategies = create_initial_strategies()
        for state, strategy, outcome in initial_strategies:
            lib.add_record(state, strategy, outcome)
        print("Initialized library with example strategies.")
    
    # 模拟一个当前状态进行检索
    current_state = TrainingState(epoch=3, avg_train_loss=1.5, grad_norm=1.0, progress=0.2, val_loss=1.6)
    recommended_strategy = lib.retrieve_best_strategy(current_state, k=2)
    
    if recommended_strategy:
        print("\n--- Retrieved Best Strategy ---")
        print(f"Learning Rate: {recommended_strategy.learning_rate}")
        print(f"Optimizer: {recommended_strategy.optimizer_name}")
        print(f"Gradient Accumulation Steps: {recommended_strategy.gradient_accumulation_steps}")
        print(f"Scheduler: {recommended_strategy.scheduler_type}")
        print(f"Scheduler Kwargs: {recommended_strategy.scheduler_kwargs}")
    else:
        print("\nNo strategy retrieved.")