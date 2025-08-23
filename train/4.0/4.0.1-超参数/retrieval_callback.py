# retrieval_callback.py

import logging
from typing import Dict, Any, Optional
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers import get_scheduler
import torch

# 假设 strategy_library.py 在同一目录下
from strategy_library import StrategyLibrary, TrainingState, Strategy

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalBasedCallback(TrainerCallback):
    """
    基于检索的回调函数，用于实现方案2.3的核心逻辑：
    1. 在每个Epoch开始时，根据当前状态检索最佳策略
    2. 应用检索到的超参数（学习率、梯度累积等）
    3. 在每个Epoch结束时，收集当前状态和效果，更新策略库
    """

    def __init__(self, strategy_library: StrategyLibrary, 
                 initial_total_epochs: int, 
                 total_training_steps: int,
                 # 可以添加其他配置，如K近邻数量等
                 k_neighbors: int = 3):
        """
        初始化回调
        :param strategy_library: 策略库实例
        :param initial_total_epochs: 计划的总训练轮数
        :param total_training_steps: 总的训练步数 (用于计算进度)
        :param k_neighbors: 检索时考虑的最近邻数量
        """
        self.strategy_library = strategy_library
        self.initial_total_epochs = initial_total_epochs
        self.total_training_steps = total_training_steps
        self.k_neighbors = k_neighbors
        
        # 用于存储上一epoch的信息，以便计算梯度范数等
        self.last_epoch_log_history = []
        
        logger.info("RetrievalBasedCallback initialized.")

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        在每个Epoch开始时调用
        1. 构造当前训练状态
        2. 调用策略库进行检索
        3. 应用推荐的策略
        """
        try:
            trainer = kwargs.get('model', None) # 通过kwargs获取trainer实例，但这可能不直接可用
            # 更可靠的方式是通过state和args获取信息
            
            # --- 1. 构造当前训练状态 ---
            current_epoch = int(state.epoch) if state.epoch is not None else 0
            
            # 从log_history中获取最近的训练指标
            avg_train_loss = self._get_recent_train_loss(state)
            grad_norm = self._estimate_grad_norm(kwargs.get('model', None))
            
            # 计算训练进度
            # state.global_step 是已完成的步数
            progress = state.global_step / self.total_training_steps if self.total_training_steps > 0 else 0.0
            
            # 从log_history中获取最近的验证集指标（如果有的话）
            val_loss = self._get_recent_val_loss(state)
            val_metric = self._get_recent_val_metric(state) # 例如 accuracy, 可根据需要调整

            current_state = TrainingState(
                epoch=current_epoch,
                avg_train_loss=avg_train_loss,
                grad_norm=grad_norm,
                progress=progress,
                val_loss=val_loss,
                val_metric=val_metric
            )

            logger.info(f"[Epoch {current_epoch}] Current State: "
                        f"Loss={avg_train_loss:.4f}, GradNorm={grad_norm:.4f}, Progress={progress:.2%}")

            # --- 2. 调用策略库进行检索 ---
            recommended_strategy: Optional[Strategy] = self.strategy_library.retrieve_best_strategy(
                current_state, k=self.k_neighbors
            )

            # --- 3. 应用推荐的策略 ---
            if recommended_strategy and hasattr(kwargs.get('train_dataloader', None), 'trainer'):
                # 注意：直接修改 trainer.args 在 on_epoch_begin 可能不会完全生效
                # 更好的方式是在 on_step_begin 或通过自定义 Trainer 控制
                
                # 为了简化，我们尝试直接修改 args，但这可能需要在特定时机
                # 或者在自定义 Trainer 的 training_step 中读取动态值
                
                # 应用学习率 (需要重新创建调度器)
                old_lr = args.learning_rate
                args.learning_rate = recommended_strategy.learning_rate
                logger.info(f"Adjusted learning rate from {old_lr} to {args.learning_rate}")
                
                # 应用梯度累积步数
                old_accum = args.gradient_accumulation_steps
                args.gradient_accumulation_steps = recommended_strategy.gradient_accumulation_steps
                logger.info(f"Adjusted gradient accumulation steps from {old_accum} to {args.gradient_accumulation_steps}")
                
                # 注意：优化器和调度器的更改比较复杂，通常在训练开始时就固定了
                # 这里我们只记录推荐的优化器和调度器，实际应用可能需要更复杂的逻辑
                # 例如，在自定义Trainer中根据strategy动态创建优化器和调度器
                
                logger.info(f"Recommended Optimizer: {recommended_strategy.optimizer_name}")
                logger.info(f"Recommended Scheduler: {recommended_strategy.scheduler_type} "
                            f"with kwargs: {recommended_strategy.scheduler_kwargs}")

                # 将推荐策略存储在state中，供其他地方（如自定义Trainer）使用
                state.recommended_strategy = recommended_strategy

            elif recommended_strategy:
                logger.info("Strategy retrieved but trainer is not fully accessible in this callback stage for direct application.")
                state.recommended_strategy = recommended_strategy
            else:
                logger.info("No strategy retrieved. Using default configuration.")
                # 可以设置一个默认策略
                # state.recommended_strategy = self._get_default_strategy()

        except Exception as e:
            logger.error(f"Error in on_epoch_begin: {e}", exc_info=True)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        在每个Epoch结束时调用
        1. 收集当前Epoch的最终状态和效果
        2. 将记录添加到策略库中
        """
        try:
            # --- 1. 收集当前Epoch的最终状态和效果 ---
            current_epoch = int(state.epoch) if state.epoch is not None else 0
            
            # 获取本epoch的平均训练loss
            avg_train_loss = self._get_epoch_train_loss(state)
            
            # 估计梯度范数 (可以使用epoch结束时的值，或者平均值)
            grad_norm = self._estimate_grad_norm(kwargs.get('model', None))
            
            progress = state.global_step / self.total_training_steps if self.total_training_steps > 0 else 0.0

            # 获取验证集指标
            val_loss = self._get_recent_val_loss(state)
            val_metric = self._get_recent_val_metric(state)

            final_state = TrainingState(
                epoch=current_epoch,
                avg_train_loss=avg_train_loss,
                grad_norm=grad_norm,
                progress=progress,
                val_loss=val_loss,
                val_metric=val_metric
            )

            # --- 2. 构造结果字典 ---
            outcome = {
                "final_train_loss": avg_train_loss,
                "final_grad_norm": grad_norm,
                "final_val_loss": val_loss if val_loss is not None else float('inf'),
                "final_val_metric": val_metric if val_metric is not None else 0.0,
                # 可以添加更多指标，如训练时间、稳定性评价等
                "notes": f"Completed epoch {current_epoch}"
            }

            # --- 3. 确定本epoch实际使用的策略 ---
            # 理想情况下，我们希望知道本epoch实际使用了什么策略
            # 这可能需要在Trainer或training_step中记录
            # 这里我们做一个简化的假设：
            # 如果在on_epoch_begin时检索到了策略，就认为本epoch使用了它
            # 否则使用默认策略
            
            # 从state中获取推荐的策略（在on_epoch_begin中设置的）
            used_strategy = getattr(state, 'recommended_strategy', None)
            
            # 如果没有推荐策略，可能需要一个默认策略
            if used_strategy is None:
                # 这里可以定义一个默认策略，或者从args中提取
                used_strategy = self._extract_strategy_from_args(args)
                logger.warning("No recommended strategy found for this epoch, using strategy extracted from args or default.")

            # --- 4. 将记录添加到策略库 ---
            if used_strategy is not None:
                self.strategy_library.add_record(final_state, used_strategy, outcome)
                logger.info(f"[Epoch {current_epoch}] Added record to strategy library.")
            else:
                logger.warning(f"[Epoch {current_epoch}] Could not add record: No strategy information available.")

            # 更新日志历史记录
            self.last_epoch_log_history = list(state.log_history)

        except Exception as e:
            logger.error(f"Error in on_epoch_end: {e}", exc_info=True)


    def _get_recent_train_loss(self, state: TrainerState) -> float:
        """获取最近的训练loss"""
        if not state.log_history:
            return 0.0
        # 查找最新的包含loss的记录
        for record in reversed(state.log_history):
            if 'loss' in record:
                return record['loss']
        return 0.0

    def _get_epoch_train_loss(self, state: TrainerState) -> float:
        """获取当前epoch的平均训练loss"""
        # 比较log_history的长度变化来判断epoch边界是不可靠的
        # 更好的方法是在on_epoch_begin记录起始点，在on_epoch_end计算区间平均值
        # 这里简化处理，仍使用最近的loss
        return self._get_recent_train_loss(state)

    def _get_recent_val_loss(self, state: TrainerState) -> Optional[float]:
        """获取最近的验证集loss"""
        if not state.log_history:
            return None
        for record in reversed(state.log_history):
            if 'eval_loss' in record:
                return record['eval_loss']
        return None

    def _get_recent_val_metric(self, state: TrainerState) -> Optional[float]:
        """获取最近的验证集指标 (例如 accuracy)"""
        if not state.log_history:
            return None
        for record in reversed(state.log_history):
            # 常见的指标名
            for metric in ['eval_accuracy', 'eval_f1', 'eval_rougeL']:
                if metric in record:
                    return record[metric]
        return None

    def _estimate_grad_norm(self, model) -> float:
        """估计模型的梯度范数"""
        if model is None:
            return 0.0
        total_norm = 0.0
        param_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        if param_count == 0:
            return 0.0
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def _extract_strategy_from_args(self, args: TrainingArguments) -> Strategy:
        """从TrainingArguments中提取当前使用的策略"""
        # 这是一个简化的提取方法，实际可能需要更多信息
        return Strategy(
            learning_rate=args.learning_rate,
            optimizer_name=getattr(args, 'optim', 'adamw_torch'), # HuggingFace的optim参数
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            scheduler_type="default", # 这部分信息在args中不直接体现
            scheduler_kwargs={}
        )

    def _get_default_strategy(self) -> Strategy:
        """获取默认策略"""
        return Strategy(
            learning_rate=1e-4,
            optimizer_name="adamw_torch",
            gradient_accumulation_steps=4,
            scheduler_type="cosine",
            scheduler_kwargs={"num_cycles": 0.5}
        )

# --- 辅助函数，用于与自定义Trainer集成 ---
def apply_strategy_to_trainer(trainer, strategy: Strategy):
    """
    (可选) 一个辅助函数，展示如何在自定义Trainer中应用检索到的策略
    这需要你使用自定义的Trainer类
    """
    # 示例：修改优化器和调度器
    # 注意：这通常在Trainer初始化或训练循环开始时进行，而不是在epoch中动态修改
    # 动态修改优化器和调度器非常复杂且容易出错
    # 更推荐的方式是在Trainer初始化时就准备好支持动态策略的逻辑
    
    # 例如，在自定义Trainer的 __init__ 或 create_optimizer_and_scheduler 中:
    """
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # 使用 state.recommended_strategy 或默认策略
        strategy = getattr(self.state, 'recommended_strategy', self.default_strategy)
        if strategy:
            self.args.learning_rate = strategy.learning_rate
            self.args.gradient_accumulation_steps = strategy.gradient_accumulation_steps
            # 重新创建优化器和调度器
            super().create_optimizer()
            self.lr_scheduler = get_scheduler(
                strategy.scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                **strategy.scheduler_kwargs
            )
    """
    pass
