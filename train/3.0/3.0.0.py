# finetune_qwen3_1.7b.py
import os
import json
from dataclasses import dataclass
from typing import Dict, Sequence, Any, Optional
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
import logging

# --- 1. 配置日志 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. 配置参数 ---
@dataclass
class FinetuneConfig:
    # 模型配置 (使用本地路径)
    model_name_or_path: str = "/home/shuzhisuo/ltq/glm4.1v_project/models/Qwen3-1.7B"
    trust_remote_code: bool = True

    # 数据配置 (保持不变)
    train_file: str = "/home/shuzhisuo/wzj/qwen-3B/train.jsonl"
    val_file: str = "/home/shuzhisuo/wzj/qwen-3B/val.jsonl"
    
    # 输出配置
    output_dir: str = "/home/shuzhisuo/wzj/qwen-3B/qwen3_1.7b_lora_a800"
    
    # LoRA 配置
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    
    # 训练配置
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    num_train_epochs: int = 3
    bf16: bool = True
    optim: str = "adamw_torch"
    
    # 调度器和早停
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    load_best_model_at_end: bool = True
    
    # 评估和保存
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 2
    logging_steps: int = 50
    logging_first_step: bool = True
    
    # 早停配置
    early_stopping_patience: int = 3
    
    # 其他
    dataloader_num_workers: int = 4
    disable_tqdm: bool = False
    report_to: Optional[str] = None

# --- 3. 自定义数据集类 ---
class SupervisedDataset(TorchDataset):
    def __init__(self, raw_data, tokenizer: AutoTokenizer, max_len: int = 1024):
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
    """使用 Qwen3 的对话模板"""
    conversations = []
    for sample in sources:
        user_input = sample['query']
        bot_response = sample['response']
        conversation = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n{bot_response}<|im_end|>"
        )
        conversations.append(conversation)

    tokenized = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=max_len,
        truncation=True,
    )

    labels = tokenized["input_ids"].clone()
    # labels[labels == tokenizer.pad_token_id] = -100

    return dict(
        input_ids=tokenized["input_ids"],
        labels=labels,
        attention_mask=tokenized["attention_mask"],
    )

# --- 4. 自定义 Trainer ---
class CustomTrainer(Trainer):
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                num_cycles=0.5
            )
        return self.lr_scheduler

# --- 5. 主函数 ---
def main():
    # 禁用 wandb 以避免网络连接问题
    os.environ["WANDB_DISABLED"] = "true"
    
    config = FinetuneConfig()
    logger.info("配置加载完成。")
    os.makedirs(config.output_dir, exist_ok=True)

    # 加载分词器和模型
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
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            device_map="auto",
        )
        logger.info("模型加载完成。")
    except Exception as e:
        logger.error(f"加载模型或分词器失败: {e}")
        raise

    # LoRA 配置
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

    # 加载数据集
    logger.info("正在加载和处理数据集...")
    def load_jsonl_data(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    train_raw_data = load_jsonl_data(config.train_file)
    val_raw_data = load_jsonl_data(config.val_file)
    logger.info(f"训练集大小: {len(train_raw_data)}, 验证集大小: {len(val_raw_data)}")

    train_dataset = SupervisedDataset(train_raw_data, tokenizer)
    val_dataset = SupervisedDataset(val_raw_data, tokenizer)
    
    # 检查数据样本
    logger.info("示例训练数据:")
    logger.info(f"Input IDs 长度: {len(train_dataset[0]['input_ids'])}")
    logger.info(f"Labels 长度: {len(train_dataset[0]['labels'])}")
    
    logger.info("数据集处理完成。")

    # 配置训练参数
    logger.info("正在配置训练参数...")
    training_args_dict = {
        "output_dir": config.output_dir,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "bf16": config.bf16,
        "optim": config.optim,
        "eval_strategy": config.eval_strategy,
        "eval_steps": config.eval_steps,
        "save_strategy": config.save_strategy,
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "load_best_model_at_end": config.load_best_model_at_end,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "logging_steps": config.logging_steps,
        "logging_first_step": config.logging_first_step,
        "dataloader_num_workers": config.dataloader_num_workers,
        "disable_tqdm": config.disable_tqdm,
        "report_to": config.report_to,  # 明确设置为 None
        "remove_unused_columns": False,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "lr_scheduler_type": config.lr_scheduler_type,
        "run_name": None,  # 避免使用 output_dir 作为 run_name
    }
    training_args = TrainingArguments(**training_args_dict)
    logger.info("训练参数配置完成。")

    # 创建 Trainer
    logger.info("正在创建 Trainer...")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)]
    )
    logger.info("Trainer 创建完成。")

    # 开始训练
    logger.info("开始训练...")
    logger.info(f"训练参数详情: {training_args}")
    train_result = trainer.train()
    logger.info("训练完成。")

    # 保存模型和训练结果
    logger.info("正在保存最终模型...")
    trainer.save_model()
    trainer.save_state()
    logger.info(f"模型和训练状态已保存至: {config.output_dir}")

    # 记录训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

if __name__ == "__main__":
    main()