import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 加载.env文件
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# 从环境变量读取配置
API_KEY = os.getenv('API_KEY', 'hApEHVmHSZm1-4NHWfPWfUy9boJ4ZngnqUlJJ3_AYErSyAMNKDAg6TSeLx7-_9NXiFzAT2LArA')
BASE_URL = os.getenv('BASE_URL', 'https://router.shengsuanyun.com/api/v1')
THINK_MODEL = os.getenv('THINK_MODEL', 'deepseek/deepseek-v3.2-think')

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

try:
    completion = client.chat.completions.create(
        model=THINK_MODEL,
        messages=[{"role": "user", "content": "Which number is larger, 9.11 or 9.8?"}],
        temperature=0.6,
        top_p=0.7,
        stream=True,
    )

    reasoning_text = ""
    answer_text = ""

    for chunk in completion:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        # 思考过程（reasoning_content）通常会率先流出
        if getattr(delta, "reasoning_content", None):
            reasoning_piece = delta.reasoning_content
            reasoning_text += reasoning_piece
            print(f"[thought] {reasoning_piece}", end="", flush=True)

        # 最终回答内容
        if delta.content is not None:
            answer_piece = delta.content
            answer_text += answer_piece
            print(answer_piece, end="", flush=True)

    print("\n\n--- 收到的思考过程 ---")
    print(reasoning_text.strip())
    print("\n--- 最终回答 ---")
    print(answer_text.strip())

except Exception as e:
    print(f"Request failed: {e}")