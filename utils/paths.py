from pathlib import Path

"""
通用路径配置模块

提供项目根目录、数据目录、输出目录等常用路径常量，避免各脚本到处写相对路径。
"""

# 项目根目录：utils 的上一级目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据目录：用于保存坐标、缓存等 JSON 数据
DATA_DIR = PROJECT_ROOT / "data"

# 输出目录：用于保存分析结果等文本 / JSON
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# 截图 / 图片输出目录
CAPTURE_DIR = OUTPUT_DIR / "captures"

# 确保目录存在
for _dir in (DATA_DIR, OUTPUT_DIR, CAPTURE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

