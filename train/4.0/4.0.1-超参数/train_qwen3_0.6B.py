# train_qwen3_0.6B_with_retrieval.py

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Sequence, Any, Optional, List, Tuple
from torch.utils.data import Dataset as TorchDataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from sklearn.metrics.pairwise import cosine_distances # 需要安装 scikit-learn

# --- 导入我们之前定义的策略库和回调 ---
# 你需要将 strategy_library.py 和 retrieval_callback.py 放在同级目录下
from strategy_library import StrategyLibrary, TrainingState, Strategy, create_initial_strategies
from retrieval_callback import RetrievalBasedCallback

# --- 1. 配置日志 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. 配置参数 ---
@dataclass
class FinetuneConfig:
    # 模型配置 (请根据你的实际路径修改)
    model_name_or_path: str = "/home/shuzhisuo/wzj/qwen3-0.6B/Qwen3-0.6B-full/qwen/Qwen3-0___6B"
    trust_remote_code: bool = True

    # 数据配置 (请根据你的实际路径修改)
    train_file: str = "/home/shuzhisuo/wzj/数据增强/11w--ds.jsonl"
    val_file: Optional[str] = None # 如果没有独立的验证集，可以为 None，脚本会自动划分
    
    # 输出配置
    output_dir: str = "./qwen3_0.6B_finetuned_with_retrieval"
    strategy_library_path: str = "./qwen3_0.6B_finetuned_with_retrieval/strategy_library.json"
    
    # LoRA 配置
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj")
    
    # 初始训练配置 (会被回调动态调整)
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4 # 初始值
    learning_rate: float = 1e-4 # 初始值
    num_train_epochs: int = 3
    bf16: bool = True
    optim: str = "adamw_torch" # 初始值
    
    # 调度器和早停
    lr_scheduler_type: str = "cosine" # 初始值
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    load_best_model_at_end: bool = False # 为避免策略匹配问题，暂时禁用
    
    # 评估和保存 (使用 eval_strategy)
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 2
    logging_steps: int = 50
    logging_first_step: bool = True
    eval_delay: int = 50
    
    # 早停配置
    early_stopping_patience: int = 3
    
    # 其他
    dataloader_num_workers: int = 2
    disable_tqdm: bool = False
    report_to: Optional[str] = None

# --- 3. 自定义数据集类 (适配你的数据格式) ---
class SupervisedDataset(TorchDataset):
    """监督微调数据集"""
    def __init__(self, raw_data, tokenizer: AutoTokenizer, max_len: int = 2048):
        super(SupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.max_len = max_len
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]], self.tokenizer, self.max_len)
        ret = {
            "input_ids": ret["input_ids"][0],
            "labels": ret["labels"][0],
            "attention_mask": ret["attention_mask"][0],
        }
        self.cached_data_dict[i] = ret
        return ret

def preprocess(sources: Sequence[Dict[str, str]], tokenizer: AutoTokenizer, max_len: int) -> Dict[str, Any]:
    """预处理数据，使用简单的 Prompt 格式"""
    conversations = []
    for sample in sources:
        # 你的数据格式: {"query": "...", "response": "..."}
        user_input = sample['query']
        bot_response = sample['response']
        
        # 使用简单的 Prompt 格式 (可以根据需要切换为 Qwen 的官方格式)
        conversation = f"### Query:\n{user_input}\n\n### Response:\n{bot_response}{tokenizer.eos_token}"
        # 或者使用 Qwen 官方格式:
        # conversation = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n{bot_response}<|im_end|>"
        
        conversations.append(conversation)
        
    tokenized = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=max_len,
        truncation=True,
    )
    
    labels = tokenized["input_ids"].clone()
    # labels[labels == tokenizer.pad_token_id] = -100 # 可选：忽略 padding token 的 loss
    
    return dict(
        input_ids=tokenized["input_ids"],
        labels=labels,
        attention_mask=tokenized["attention_mask"],
    )

# --- 4. 自定义 Trainer 以更好地与回调集成 ---
class RetrievalAwareTrainer(Trainer):
    """
    一个可以感知检索到的策略并应用它的 Trainer
    注意：动态修改学习率和优化器在 Trainer 中比较复杂，
    更稳健的做法是在 Callback 中记录推荐策略，然后在这里读取。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # 重写 create_scheduler 以使用自定义的余弦退火调度器 (与原脚本一致)
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                num_cycles=0.5 # 与训练方法.txt中的推荐一致
            )
        return self.lr_scheduler

# --- 5. 主函数 ---
def main():
    # 1. 加载配置
    config = FinetuneConfig()
    logger.info("配置加载完成。")

    # 2. 确保输出目录存在
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 3. 加载分词器和模型
    logger.info(f"正在加载模型和分词器: {config.model_name_or_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("分词器加载完成。")

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=torch.bfloat16 if config.bf16 else (torch.float16 if not config.bf16 else torch.float32),
            device_map="auto",
        )
        model.config.use_cache = False # 微调时需要关闭
        logger.info("模型加载完成。")

    except Exception as e:
        logger.error(f"加载模型或分词器失败: {e}")
        raise

    # 4. 配置 LoRA
    logger.info("正在配置 LoRA...")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logger.info("LoRA 配置完成。")

    # 5. 加载和处理数据集
    logger.info("正在加载和处理数据集...")
    def load_jsonl_data(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    train_raw_data = load_jsonl_data(config.train_file)
    
    # 划分训练集和验证集
    if config.val_file and os.path.exists(config.val_file):
        val_raw_data = load_jsonl_data(config.val_file)
    else:
        # 如果没有独立的验证集，则从训练集中划分
        logger.info("未提供独立验证集，从训练集中自动划分...")
        split_idx = int(0.95 * len(train_raw_data)) # 95% 用于训练
        val_raw_data = train_raw_data[split_idx:]
        train_raw_data = train_raw_data[:split_idx]
        
    logger.info(f"训练集大小: {len(train_raw_data)}, 验证集大小: {len(val_raw_data)}")

    train_dataset = SupervisedDataset(train_raw_data, tokenizer)
    val_dataset = SupervisedDataset(val_raw_data, tokenizer)
    logger.info("数据集处理完成。")

    # 6. 初始化策略库
    logger.info("正在初始化策略库...")
    strategy_library = StrategyLibrary(library_path=config.strategy_library_path)
    if not strategy_library.records:
        logger.info("策略库为空，初始化示例策略...")
        initial_strategies = create_initial_strategies()
        for state, strategy, outcome in initial_strategies:
            strategy_library.add_record(state, strategy, outcome)
    logger.info(f"策略库初始化完成，包含 {len(strategy_library.records)} 条记录。")

    # 7. 配置训练参数
    logger.info("正在配置训练参数...")
    
    # 计算总步数，用于回调
    total_train_steps = (
        len(train_dataset) // 
        (config.per_device_train_batch_size * config.gradient_accumulation_steps)
    ) * config.num_train_epochs

    training_args_dict = {
        "output_dir": config.output_dir,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps, # 初始值
        "learning_rate": config.learning_rate, # 初始值
        "num_train_epochs": config.num_train_epochs,
        "bf16": config.bf16,
        "optim": config.optim, # 初始值
        "eval_strategy": config.eval_strategy,
        "eval_steps": config.eval_steps,
        "save_strategy": config.save_strategy,
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "load_best_model_at_end": config.load_best_model_at_end, # 已禁用
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "logging_steps": config.logging_steps,
        "logging_first_step": config.logging_first_step,
        "eval_delay": config.eval_delay,
        "dataloader_num_workers": config.dataloader_num_workers,
        "disable_tqdm": config.disable_tqdm,
        "report_to": config.report_to,
        "remove_unused_columns": False,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "lr_scheduler_type": config.lr_scheduler_type, # 初始值
    }
    
    training_args = TrainingArguments(**training_args_dict)
    logger.info("训练参数配置完成。")

    # 8. 创建 Trainer 和 Callback
    logger.info("正在创建 Trainer 和 Callback...")
    
    # 初始化回调
    retrieval_callback = RetrievalBasedCallback(
        strategy_library=strategy_library,
        initial_total_epochs=config.num_train_epochs,
        total_training_steps=total_train_steps,
        k_neighbors=3
    )
    
    trainer = RetrievalAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[retrieval_callback, EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)]
    )
    logger.info("Trainer 和 Callback 创建完成。")

    # 9. 开始训练
    logger.info("开始训练...")
    train_result = trainer.train()
    logger.info("训练完成。")

    # 10. 保存模型、状态和最终的策略库
    logger.info("正在保存最终模型、状态和策略库...")
    trainer.save_model()
    trainer.save_state()
    strategy_library.save_library() # 保存训练过程中更新的策略库
    logger.info(f"模型、状态和策略库已保存至: {config.output_dir}")

    # 11. 记录训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # 可选：评估最终模型
    logger.info("运行最终评估...")
    eval_results = trainer.evaluate()
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)
    logger.info("最终评估完成。")

if __name__ == "__main__":
    main()