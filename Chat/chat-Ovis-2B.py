"""
一个用于与本地 Ovis2.5-2B 模型进行多轮对话的命令行脚本。
支持在每轮对话中可选地提供一张图片。
"""

import os
from PIL import Image
import torch
from transformers import AutoModelForCausalLM

# --- 1. 配置模型路径 ---
MODEL_PATH = "/home/balcony/models/Ovis2.5-2B"

# --- 2. 检查模型路径 ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型未找到，请检查路径: {MODEL_PATH}")

# --- 3. 加载模型 ---
print("正在加载 Ovis2.5-2B 模型，请稍候...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"  # 自动选择设备 (GPU优先)
).eval()
print("模型加载成功！\n")

# --- 4. 定义对话历史 ---
# Ovis 是一个对话模型，需要维护对话历史以实现上下文理解
conversation_history = []

def get_user_input():
    """获取用户输入的文本和可选的图片路径"""
    print("-" * 30)
    
    # 1. 询问是否使用图片
    use_image = input("本轮对话需要图片吗? (y/N 或直接回车跳过): ").strip().lower()
    image = None
    if use_image in ['y', 'yes']:
        image_path = input("请输入图片的绝对路径 (e.g., /home/balcony/test.jpg): ").strip()
        if not os.path.exists(image_path):
            print(f"警告: 图片路径 '{image_path}' 不存在，本轮将不使用图片。")
        else:
            try:
                image = Image.open(image_path).convert("RGB")
                print("图片加载成功。")
            except Exception as e:
                print(f"警告: 加载图片时出错: {e}，本轮将不使用图片。")

    # 2. 获取文本输入
    text_input = input("请输入你的问题 (输入 'exit' 或 'quit' 退出): ").strip()
    if text_input.lower() in ['exit', 'quit']:
        return None, None # 信号退出
    
    return image, text_input

def run_inference(image, text):
    """使用模型进行单轮推理"""
    try:
        # --- 准备输入内容 ---
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": text})
        
        # 将当前轮次的用户消息添加到历史中
        conversation_history.append({"role": "user", "content": content})

        # --- 预处理输入 ---
        preprocess_output = model.preprocess_inputs(messages=conversation_history, add_generation_prompt=True)
        input_ids = preprocess_output[0].to(model.device)
        pixel_values = preprocess_output[1].to(model.device, dtype=torch.bfloat16) if preprocess_output[1] is not None else None
        grid_thws = preprocess_output[2].to(model.device) if len(preprocess_output) > 2 and preprocess_output[2] is not None else None

        # --- 执行推理 ---
        with torch.inference_mode():
            gen_kwargs = {
                "max_new_tokens": 512,
                "do_sample": False, # 使用确定性解码以获得更一致的结果
                "eos_token_id": model.text_tokenizer.eos_token_id,
                "pad_token_id": model.text_tokenizer.pad_token_id,
                "use_cache": True
            }
            generated_ids = model.generate(
                input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                **gen_kwargs
            )

        # --- 解码模型输出 ---
        response_ids = generated_ids[0]
        full_response = model.text_tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # 将模型的回答也添加到历史中，以便下一轮对话使用上下文
        conversation_history.append({"role": "assistant", "content": full_response})

        return full_response

    except Exception as e:
        print(f"模型推理时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return "抱歉，处理您的请求时出现了内部错误。"

# --- 5. 主循环 ---
if __name__ == "__main__":
    print("欢迎使用 Ovis2.5-2B 对话系统!")
    print("提示: 在每轮对话开始时，您可以选择是否提供一张图片。")
    print("输入 'exit' 或 'quit' 可以退出程序。\n")

    while True:
        image, user_text = get_user_input()
        
        # 检查是否收到退出信号
        if user_text is None:
            print("再见!")
            break

        print("\n正在思考...")
        response = run_inference(image, user_text)
        
        print("\n[模型回答]")
        print(response)
        print("\n")
