# test_lora_model.py
import torch
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel # 需要导入 PeftModel
from qwen_vl_utils import process_vision_info
import os
import argparse

def load_lora_model(base_model_path, lora_model_path):
    """
    加载基座模型和 LoRA 权重。
    """
    print(f"Loading base model from {base_model_path}...")
    # 1. 加载基座模型
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    print("Base model loaded.")

    print(f"Loading LoRA weights from {lora_model_path}...")
    # 2. 加载 LoRA 权重到基座模型
    # PeftModel.from_pretrained 会自动查找 lora_model_path 下的 adapter_config.json 和权重文件
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    print("LoRA model loaded.")
    return model

def load_tokenizer_and_processor(model_path):
    """
    加载 tokenizer 和 processor。
    """
    print(f"Loading tokenizer and processor from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("Tokenizer and processor loaded.")
    return tokenizer, processor

def predict(messages, model, processor, tokenizer, max_new_tokens=200):
    """
    使用模型进行预测。
    """
    # 准备推理输入
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # 将输入移动到模型所在的设备
    inputs = {k: v.to(model.device) for k, v in inputs.items() if v is not None}

    # 生成输出
    with torch.no_grad(): # 禁用梯度计算以节省资源
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    # 去除输入部分，只保留生成的部分
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
    ]
    # 解码生成的 token 为文本
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0] # 返回第一个（也是唯一一个）生成结果

def run_test(model, tokenizer, processor):
    """
    运行测试循环。
    """
    print("\n" + "="*20 + " 模型测试 " + "="*20)
    print("输入 'quit' 或 'exit' 退出测试。")
    print("输入 'text' 开始纯文本对话模式。")
    print("输入图片路径 (例如 /path/to/image.jpg) 开始图文对话模式。")
    print("-" * 50)

    current_mode = "image" # 初始模式可以是 image 或 text
    image_path = None

    while True:
        try:
            # 获取用户输入
            user_input = input("\n[User Input] ").strip()

            if user_input.lower() in ['quit', 'exit']:
                print("退出测试。")
                break

            if user_input.lower() == 'text':
                current_mode = "text"
                image_path = None # 清除图片路径
                print("已切换到纯文本对话模式。请输入你的问题。")
                # 询问用户问题
                question = input("[Your Question] ").strip()
                if not question:
                    print("未输入问题。")
                    continue
                user_input = question # 将问题作为实际的用户输入

            if os.path.isfile(user_input): # 检查输入是否为有效文件路径
                current_mode = "image"
                image_path = user_input
                print(f"已切换到图文对话模式，使用图片: {image_path}")
                # 询问用户关于这张图片的问题
                question = input("[Question about the image] ").strip()
                if not question:
                    print("未输入问题。")
                    continue
                user_input = question # 将问题作为实际的用户输入

            if not user_input:
                print("输入不能为空，请重新输入。")
                continue

            # 构造 messages
            if current_mode == "image" and image_path and os.path.exists(image_path):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_path, # 使用用户提供的图片路径
                            },
                            {
                                "type": "text",
                                "text": user_input, # 用户的文本输入或关于图片的问题
                            }
                        ]
                    }
                ]
            else:
                # 纯文本模式或图片路径无效时
                messages = [
                    {
                        "role": "user",
                        "content": user_input # 纯文本输入
                    }
                ]
            
            print("[Model is thinking...]")
            response = predict(messages, model, processor, tokenizer)
            
            print(f"[Model Response]\n{response}")

        except KeyboardInterrupt:
            print("\n\n收到中断信号，退出测试。")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            print("请检查输入或模型状态，然后重试。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the trained Qwen2.5-VL LoRA model.")
    # 基座模型路径 (你的原始模型路径)
    parser.add_argument("--base_model_path", type=str, 
                        default="/home/shuzhisuo/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-3B-Instruct",
                        help="Path to the base Qwen2.5-VL model directory.")
    # LoRA模型路径 (你训练脚本的 output_dir)
    parser.add_argument("--lora_model_path", type=str, 
                        default="./output/Qwen2.5-VL-3B-700+300",
                        help="Path to the trained LoRA model directory (output_dir from training).")
    
    args = parser.parse_args()

    try:
        # 1. 加载 tokenizer 和 processor (通常与 base model 或 lora model path 相同)
        # 这里我们从 lora model path 加载，因为它也被保存了
        tokenizer, processor = load_tokenizer_and_processor(args.lora_model_path)
        
        # 2. 加载基座模型和 LoRA 权重
        model = load_lora_model(args.base_model_path, args.lora_model_path)
        
        # 3. 运行测试
        run_test(model, tokenizer, processor)
        
    except Exception as e:
        print(f"启动测试时发生错误: {e}")
