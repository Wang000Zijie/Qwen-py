#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# qwen_chat.py - Qwen3 本地对话脚本（Linux 专用） - 已启用 FlashAttention-2 + 编译缓存

import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
import os
import threading
import time

# ==================== 🔧 可配置参数 ====================
# 修改为你的实际模型路径
model_path = "/home/balcony/models/Qwen3-1.7B"

# 对话保存目录（可选）
save_dir = "/home/balcony/Qwen/cb"
os.makedirs(save_dir, exist_ok=True)

# 生成参数（Qwen 官方推荐）
temperature      = 0.7
top_p            = 0.8
top_k            = 20
min_p            = 0.0
presence_penalty = 1.1
max_new_tokens   = 1024           # ✅ 可设任意值：256, 512, 768, 1024...
max_history_rounds = 2           # ✅ 减少历史轮数，提升速度

# 系统提示语
system_prompt = "你是一个有帮助的AI助手。请用中文回答，保持专业和友好。"
# ======================================================

# 检查模型路径
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ 模型路径不存在: {model_path}")

print("🔍 检查模型路径:", model_path)
print("📂 模型目录内容:", os.listdir(model_path)[:3])

# ==================== 加载模型 ====================
print("🔄 正在加载模型...")

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True  # 强制本地加载
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",                    # 强制使用 GPU
    attn_implementation="flash_attention_2", # 🔥 启用 FlashAttention-2
    offload_folder=None,                    # 禁用 offload
    local_files_only=True
).eval()

# ✅ 编译模型 + 启用磁盘缓存（首次慢，后续飞快）
compile_cache_dir = "/home/balcony/.cache/torch_compile"
os.makedirs(compile_cache_dir, exist_ok=True)

try:
    print("🔥 编译模型中（首次运行稍慢，后续将从缓存加载）...")
    model = torch.compile(
        model,
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=False,
        cache_dir=compile_cache_dir  # ✅ 关键：编译结果存硬盘
    )
    print("✅ 模型编译完成，后续运行将显著加速")
except Exception as e:
    print(f"⚠️ 编译失败（可忽略）: {e}")

print("✅ 模型加载完成，开始对话（输入 'exit' 退出）\n")

# ==================== 对话历史 ====================
history = []

def stream_generate(input_ids):
    """流式生成函数"""
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=10.0
    )
    gen_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "repetition_penalty": presence_penalty,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    return streamer

# ==================== 主循环 ====================
while True:
    try:
        user_input = input("User: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 再见！")
            break

        # ✅ 限制历史轮数，防止上下文爆炸
        recent_history = history[-max_history_rounds:]

        # 构建消息
        messages = [{"role": "system", "content": system_prompt}]
        for u, a in recent_history:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user_input})

        # 应用模板
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 流式输出
        print("Assistant: ", end="", flush=True)
        start_time = time.time()
        streamer = stream_generate(inputs.input_ids)

        response = ""
        token_count = 0
        for new_text in streamer:
            print(new_text, end="", flush=True)
            response += new_text
            token_count += 1  # 粗略计数
        print()

        # ✅ 显示生成速度
        if token_count > 0:
            duration = time.time() - start_time
            speed = token_count / duration
            print(f"⏱️ 生成 {token_count} 个 token，耗时 {duration:.2f}s → {speed:.2f} token/s")

        # 保存对话
        history.append((user_input, response))
        if len(history) > 10:
            history = history[-10:]

    except KeyboardInterrupt:
        print("\n❌ 生成被中断")
    except torch.cuda.OutOfMemoryError:
        print("\n❌ 显存不足！自动清理缓存...")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"\n❌ 错误: {e}")