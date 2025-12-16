# download_model.py
from modelscope import snapshot_download
import os

# ================== 配置信息 ==================
model_id = 'Qwen/Qwen3-VL-4B-Instruct'  # 通义千问3 视觉语言模型 4B 指令版
save_path = '/home/balcony/models/Qwen3-VL-4B-Instruct'  # 保存路径
revision = 'master'  # 可指定版本，如 'v1.0.0'，默认最新
# =============================================

print(f"🚀 开始从魔塔（ModelScope）下载 {model_id} 到 {save_path} ...")
os.makedirs(save_path, exist_ok=True)

try:
    # 使用 snapshot_download 完整下载模型
    model_dir = snapshot_download(
        model_id=model_id,
        cache_dir=save_path,
        revision=revision,
        ignore_file_pattern=[]  # 不忽略任何文件（完整下载）
    )
    print(f"✅ 模型已成功下载到: {model_dir}")
    print("💡 提示：该模型为 Qwen3-VL-4B-Instruct，支持视觉语言多模态任务。")
    print("💡 可用于图像理解、视觉问答、文档分析等多种应用场景。")
except Exception as e:
    print(f"❌ 下载失败: {e}")