# 魔法工艺自动化项目

一个基于计算机视觉和AI的魔法工艺游戏自动化系统，支持实时检测、法术构建、门选择等核心功能。

## 📋 目录

- [功能特性](#功能特性)
- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [配置说明](#配置说明)
- [使用方法](#使用方法)
- [项目结构](#项目结构)
- [常见问题](#常见问题)

## ✨ 功能特性

### 核心功能

1. **实时检测 (realtime_detect.py)**
   - 使用YOLOv5模型实时检测游戏窗口
   - 支持动态调整每个分类的检测阈值
   - 自动瞄准、闪避、攻击等控制功能

2. **法术构建 (spell_construct_flow.py)**
   - 智能分析和构建法术配置
   - 自动识别危险法术并避免使用
   - 支持多法杖配置管理

3. **门选择 (door_handler.py)**
   - 自动分析门场景
   - 智能选择最优门选项
   - 集成法术分析流程

4. **选项选择 (option_selector.py)**
   - 使用OCR和多模态AI识别游戏选项
   - 智能决策选择最优选项

## 🔧 环境要求

- Python 3.8+
- Windows 10/11（游戏运行环境）
- CUDA支持的GPU（推荐，用于YOLO模型加速）
- 魔法工艺游戏（需要运行在本地）

## 📦 安装步骤

### 1. 克隆或下载项目

```bash
# 如果使用git
git clone <repository-url>
cd train_MAGIC
```

### 2. 创建虚拟环境（推荐）

```bash
# 使用venv
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制 `.env.example` 文件为 `.env`：

```bash
# Windows PowerShell
Copy-Item .env.example .env

# Windows CMD
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

然后编辑 `.env` 文件，填写你的实际配置（见下方配置说明）。

### 5. 准备YOLO模型文件

你需要准备训练好的YOLO模型文件：

1. **模型权重文件** (`.pt` 文件)
   - 将模型文件放在项目目录下的 `weights/` 文件夹
   - 或修改 `.env` 中的 `MODEL_WEIGHTS_PATH` 指向你的模型路径

2. **数据配置文件** (`.yaml` 文件)
   - 包含类别名称等配置信息
   - 将文件放在项目根目录
   - 或修改 `.env` 中的 `MODEL_DATA_YAML` 指向你的配置文件路径

## ⚙️ 配置说明

### 环境变量配置 (.env文件)

项目使用 `.env` 文件管理所有敏感配置。以下是所有可配置项：

#### API配置

```env
# 阿里云百炼API密钥（用于AI模型调用）
API_KEY=your_api_key_here

# API基础URL
BASE_URL=https://router.shengsuanyun.com/api/v1
```

#### 模型配置

```env
# 多模态模型（用于图像识别和分析）
MULTIMODAL_MODEL=ali/qwen3-vl-plus

# 文本模型（用于决策和文本分析）
TEXT_MODEL=ali/qwen3-max

# 思考模型（用于复杂推理任务）
THINK_MODEL=deepseek/deepseek-v3.2-think
```

#### 游戏API端点配置

```env
# 法术API端点（游戏内服务地址）
SPELLS_ENDPOINT=http://localhost:1234/spells

# 商店API端点（游戏内服务地址）
STORE_ENDPOINT=http://localhost:1234/store
```

#### YOLO模型配置

```env
# 模型权重文件路径
# 可以使用相对路径（相对于项目根目录）或绝对路径
MODEL_WEIGHTS_PATH=weights/best.pt

# 数据配置文件路径
# 可以使用相对路径（相对于项目根目录）或绝对路径
MODEL_DATA_YAML=data.yaml
```

### 获取API密钥

1. **阿里云百炼API**
   - 访问 [阿里云百炼平台](https://bailian.console.aliyun.com/)
   - 注册/登录账号
   - 创建API密钥
   - 将密钥填入 `.env` 文件的 `API_KEY`

2. **游戏API端点**
   - 确保游戏内服务正在运行
   - 默认地址为 `http://localhost:1234`
   - 如果端口不同，请修改 `.env` 中的对应配置

## 🚀 使用方法

### 实时检测

运行实时检测脚本：

```bash
python realtime_detect.py
```

**功能说明：**
- 自动检测游戏窗口
- 实时显示检测结果
- 支持按键调整检测阈值
- 自动控制功能（瞄准、闪避、攻击）

**按键控制：**
- `Q` - 退出程序
- `T` - 切换显示模式
- 其他按键根据代码中的配置

### 法术构建

```bash
python spell_construct_flow.py
```

**功能说明：**
- 分析当前法术配置
- 智能构建最优法术组合
- 自动识别并避免危险法术

### 门选择

门选择功能集成在 `realtime_detect.py` 中，当检测到门时会自动触发。

### 选项选择

```bash
python option_selector.py
```

**功能说明：**
- 截取游戏窗口
- 使用OCR识别选项
- AI智能选择最优选项

## 📁 项目结构

```
train_MAGIC/
├── .env                    # 环境变量配置（需要自己创建）
├── .env.example           # 环境变量模板
├── requirements.txt       # Python依赖包
├── README.md              # 本文件
│
├── realtime_detect.py     # 实时检测主程序
├── spell_construct_flow.py # 法术构建
├── door_handler.py        # 门处理逻辑
├── option_selector.py     # 选项选择
│
├── get_game_window.py     # 游戏窗口捕获
├── capture_and_analyze.py # 捕获和分析
├── store_auto_flow.py    # 商店自动流程
├── store_purchase_decision.py # 商店购买决策
├── mark_spell_slots.py   # 标记法术槽位
│
├── auto_attack.py         # 自动攻击
├── auto_dodge.py          # 自动闪避
├── control_thread.py      # 控制线程
│
├── omni_models/           # AI模型模块
│   ├── omni.py           # 多模态和文本模型
│   └── think_model.py    # 思考模型
│
├── utils/                 # 工具函数
│   ├── paths.py          # 路径配置
│   └── logger.py         # 日志工具
│
├── weights/              # YOLO模型权重（需要自己准备）
│   └── best.pt
│
├── data.yaml             # YOLO数据配置（需要自己准备）
│
└── *.json                # 各种配置文件
    ├── slots_*.json      # 法术槽位配置
    ├── spell_slot.json   # 法术槽位信息
    └── store_*.json      # 商店缓存文件
```

## ❓ 常见问题

### 1. 找不到游戏窗口

**问题：** 运行程序时提示"无法找到Magicraft游戏窗口"

**解决方案：**
- 确保游戏正在运行
- 检查游戏窗口标题是否包含"Magicraft"或"魔法工艺"
- 尝试以管理员权限运行程序

### 2. 模型文件不存在

**问题：** 提示"模型文件不存在"

**解决方案：**
- 检查 `.env` 文件中的 `MODEL_WEIGHTS_PATH` 和 `MODEL_DATA_YAML` 配置
- 确保模型文件路径正确
- 如果使用相对路径，确保文件在项目目录下

### 3. API调用失败

**问题：** AI模型调用失败，返回错误

**解决方案：**
- 检查 `.env` 文件中的 `API_KEY` 是否正确
- 确认网络连接正常
- 检查API配额是否用完
- 验证 `BASE_URL` 是否正确

### 4. 游戏API端点连接失败

**问题：** 无法连接到 `http://localhost:1234`

**解决方案：**
- steam右键游戏 -> 属性 -> 测试版 输入AIBetaTest1107。  进入游戏后按~呼出控制台，输入aidata_server开启端口
- 确保游戏内服务正在运行
- 检查端口号是否正确（默认1234）
- 如果端口不同，修改 `.env` 中的 `SPELLS_ENDPOINT` 和 `STORE_ENDPOINT`

### 5. 依赖包安装失败

**问题：** `pip install -r requirements.txt` 失败

**解决方案：**
- 确保Python版本 >= 3.8
- 尝试升级pip: `python -m pip install --upgrade pip`
- 对于PyTorch，可能需要根据你的CUDA版本安装：
  ```bash
  # CPU版本
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  
  # CUDA版本（根据你的CUDA版本选择）
  pip install torch --index-url https://download.pytorch.org/whl/cu118
  ```

### 6. 检测精度不高

**问题：** YOLO模型检测结果不准确

**解决方案：**
- 调整 `realtime_detect.py` 中各类别的阈值
- 使用训练更好的模型
- 检查游戏窗口是否被正确捕获

## 🔒 安全提示

- **不要将 `.env` 文件提交到版本控制系统**
- `.env` 文件包含敏感信息（API密钥等）
- 建议将 `.env` 添加到 `.gitignore`

## 📝 开发说明

### 添加新的环境变量

1. 在 `.env.example` 中添加新配置项
2. 在代码中使用 `os.getenv('YOUR_VAR', 'default_value')` 读取
3. 更新本README的配置说明部分

### 代码结构

- 核心功能模块：`realtime_detect.py`, `spell_construct_flow.py` 等
- AI模型模块：`omni_models/` 目录
- 工具函数：`utils/` 目录
- 配置文件：JSON格式的配置文件

## 📄 许可证

[根据实际情况填写]

## 👥 贡献

欢迎提交Issue和Pull Request！

---

**注意：** 本项目仅供学习和研究使用，请遵守游戏服务条款。

