# qwen_simple_chat.py
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
from threading import Thread
import os
from datetime import datetime
import subprocess
import re

# 模型路径
model_path = r"E:\Qwen\Qwen3-0.6B-full\qwen\Qwen3-0___6B"

# 加载 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
).eval()

print("✅ Qwen3-0.6B 模型加载完成，开始对话（输入 'exit' 退出）...")
print("💡 命令说明:")
print("   /yes  - 开启思考模式")
print("   /no   - 关闭思考模式")
print("   /help - 显示帮助信息")
print("📝 默认关闭思考模式\n")

# 初始化设置
thinking_enabled = False
system_prompt = "你是一个有帮助的AI助手。请用中文回答用户的问题，保持友好和专业。"
history = []

def show_help():
    """显示帮助信息"""
    print("\n📋 可用命令:")
    print("  /yes  - 开启思考模式")
    print("  /no   - 关闭思考模式")
    print("  /help - 显示帮助信息")
    print("  /save - 保存对话历史到文件")
    print("  /exit - 退出程序")
    print("  /run  - 运行外部脚本（格式：/run 脚本路径）")
    print()

def save_conversation():
    """保存对话历史到文件"""
    filename = "talking0.6.txt"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"对话保存时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            for i, (user_msg, assistant_msg) in enumerate(history, 1):
                f.write(f"[第{i}轮对话]\n")
                f.write(f"User: {user_msg}\n")
                f.write(f"Assistant: {assistant_msg}\n")
                f.write("-" * 30 + "\n")
            
        print(f"✅ 对话已保存到: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")

def run_external_script(script_path):
    """运行外部脚本并返回输出"""
    try:
        # 清理路径中的引号
        script_path = script_path.strip().strip('"').strip("'")
        
        # 检查文件是否存在
        if not os.path.exists(script_path):
            return f"❌ 脚本文件不存在: {script_path}"
        
        # 根据文件类型选择运行方式
        if script_path.endswith('.py'):
            # 运行Python脚本
            result = subprocess.run([sys.executable, script_path], 
                                  capture_output=True, text=True, encoding='utf-8')
        elif script_path.endswith('.bat'):
            # 运行批处理文件
            result = subprocess.run([script_path], 
                                  capture_output=True, text=True, encoding='utf-8')
        else:
            return f"❌ 不支持的文件类型: {script_path}"
        
        # 组合输出结果
        output = ""
        if result.stdout:
            output += f"标准输出:\n{result.stdout}\n"
        if result.stderr:
            output += f"错误输出:\n{result.stderr}\n"
        if result.returncode != 0:
            output += f"返回码: {result.returncode}\n"
        
        return output.strip() if output else "✅ 脚本执行完成，无输出"
        
    except Exception as e:
        return f"❌ 执行脚本时出错: {e}"

def stream_output(inputs, max_new_tokens=512):
    """流式输出生成函数"""
    full_response = ""
    print("Assistant: ", end="", flush=True)
    
    with torch.no_grad():
        from transformers import TextIteratorStreamer
        
        streamer = TextIteratorStreamer(
            tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        # 根据思考模式使用不同的生成参数
        if thinking_enabled:
            # 思考模式参数
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,  # 思考模式推荐温度
                top_p=0.95,       # 思考模式推荐top-p
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer
            )
        else:
            # 非思考模式参数
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,  # 非思考模式推荐温度
                top_p=0.8,        # 非思考模式推荐top-p
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer
            )
        
        # 在单独线程中启动生成
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # 逐词输出
        for new_text in streamer:
            print(new_text, end="", flush=True)
            full_response += new_text
            
    print()  # 换行
    return full_response

# 显示初始帮助信息
show_help()

while True:
    user_input = input("User: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        # 退出前询问是否保存
        save_choice = input("退出前是否保存对话历史？(y/n): ").strip().lower()
        if save_choice == 'y':
            save_conversation()
        print("再见！")
        break
    
    # 检查是否为特殊命令
    if user_input == '/yes':
        thinking_enabled = True
        print("🤔 思考模式已开启")
        continue
    elif user_input == '/no':
        thinking_enabled = False
        print("🚫 思考模式已关闭")
        continue
    elif user_input == '/help':
        show_help()
        continue
    elif user_input == '/save':
        save_conversation()
        continue
    elif user_input.startswith('/run '):
        # 运行外部脚本
        script_path = user_input[5:].strip()
        print(f"🔄 正在运行脚本: {script_path}")
        script_output = run_external_script(script_path)
        print(f"📋 脚本输出:\n{script_output}")
        # 将脚本输出作为用户输入继续对话
        user_input = f"我运行了脚本 {script_path}，输出如下：\n{script_output}"
    elif user_input.startswith('/'):
        print("❌ 未知命令，输入 /help 查看可用命令")
        continue
    
    # 检查是否包含"调用"关键词
    call_match = re.search(r'([A-Za-z]:\\[^+]*?\.(py|bat|exe))\+调用', user_input)
    if call_match:
        script_path = call_match.group(1)
        print(f"🔄 检测到脚本调用: {script_path}")
        script_output = run_external_script(script_path)
        print(f"📋 脚本输出:\n{script_output}")
        # 将脚本输出作为用户输入继续对话
        user_input = f"我运行了脚本 {script_path}，输出如下：\n{script_output}"
    
    # 构建对话消息
    messages = [{"role": "system", "content": system_prompt}]
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_input})

    try:
        # 关键：在apply_chat_template中设置enable_thinking
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking_enabled  # 这里控制是否启用思考
        )
        
        # 编码输入
        model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # 流式输出
        response = stream_output(model_inputs)
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        response = "抱歉，我遇到了一个问题。"
        print(f"Assistant: {response}")
    
    # 更新历史
    history.append((user_input, response))
    if len(history) > 10:  # 保留最近10轮对话
        history = history[-10:]