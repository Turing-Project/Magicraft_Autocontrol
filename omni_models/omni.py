"""
阿里云百炼API配置模块

统一管理多模态模型和文本模型的配置。
"""

import os
import base64
import io
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI
from PIL import Image

# 加载.env文件
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# API配置（从环境变量读取）
API_KEY = os.getenv('API_KEY', 'hApEHVmHSZm1-4NHWfPWfUy9boJ4ZngnqUlJJ3_AYErSyAMNKDAg6TSeLx7-_9NXiFzAT2LArA')
BASE_URL = os.getenv('BASE_URL', 'https://router.shengsuanyun.com/api/v1')

# 模型配置（从环境变量读取）
MULTIMODAL_MODEL = os.getenv('MULTIMODAL_MODEL', 'ali/qwen3-vl-plus')  # 多模态模型（用于图像识别）
TEXT_MODEL = os.getenv('TEXT_MODEL', 'ali/qwen3-max')  # 文本模型（用于决策和文本分析）

# 创建统一的客户端实例
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


def get_multimodal_client():
    """获取多模态模型客户端"""
    return client


def get_text_client():
    """获取文本模型客户端"""
    return client


def get_multimodal_model():
    """获取多模态模型名称"""
    return MULTIMODAL_MODEL


def get_text_model():
    """获取文本模型名称"""
    return TEXT_MODEL


def compress_image(image_path, max_size=(1024, 1024), quality=85):
    """压缩图片，减小文件大小"""
    try:
        img = Image.open(image_path)

        # 转换为RGB模式（如果是RGBA/调色板等）
        if img.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # 按比例缩放，保持宽高比
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

        # 保存到内存
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=quality, optimize=True)
        output.seek(0)
        return output
    except Exception as e:
        print(f"压缩图片失败: {e}，使用原图")
        return None


def encode_image(image_path, compress=True, max_size=(1024, 1024), quality=85):
    """将图片编码为base64格式，可选择压缩"""
    if compress:
        compressed = compress_image(image_path, max_size, quality)
        if compressed:
            return base64.b64encode(compressed.read()).decode("utf-8")

    # 如果压缩失败或不压缩，使用原图
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


if __name__ == "__main__":
    print(get_multimodal_model())
    print(get_text_model())
    img_path = r"C:\Users\sjc\Desktop\train_MAGIC\spell_slots_marked.png"
    res = encode_image(img_path)
    model_res = client.chat.completions.create(
        model=get_multimodal_model(),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请分析这个游戏画面，第一行十个法术槽有几个法术？。不要使用markdown语法。以json格式返回"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{res}"
                        }
                    }
                ]
            }
        ]
    )
    print(model_res.choices[0].message.content)