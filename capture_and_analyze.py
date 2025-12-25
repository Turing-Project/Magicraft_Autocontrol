import base64
import time

import cv2
import numpy as np

from get_game_window import GameWindowCapture
from omni_models.omni import get_multimodal_client, get_multimodal_model
from utils.paths import CAPTURE_DIR, OUTPUT_DIR

class GameAnalyzer:
    """游戏画面分析与AI解析器"""
    
    def __init__(self):
        # 使用统一的多模态模型配置
        self.client = get_multimodal_client()
        self.model = get_multimodal_model()
        
    def capture_magicraft_screen(self):
        """捕获Magicraft游戏窗口"""
        capturer = GameWindowCapture()
        
        if not capturer.select_magicraft_window():
            print("✗ 无法找到Magicraft游戏窗口")
            return None
        
        # 捕获一帧画面
        frame = capturer.capture_frame_win32api()
        return frame
    
    def crop_region(self, frame):
        """截取左上角300高1000宽的图片"""
        if frame is None:
            return None
        
        height, width = frame.shape[:2]
        print(f"原始画面尺寸: {width}x{height}")
        
        # 截取左上角区域 (x: 0-1000, y: 0-300)
        crop_width = min(1000, width)
        crop_height = min(300, height)
        
        cropped = frame[0:crop_height, 0:crop_width]
        print(f"截取区域尺寸: {crop_width}x{crop_height}")
        
        return cropped
    
    def image_to_base64(self, image):
        """将OpenCV图像转换为base64编码"""
        # 将BGR转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 编码为JPEG
        _, buffer = cv2.imencode('.jpg', image_rgb)
        
        # 转换为base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    
    def analyze_image(self, image_base64, prompt="请分析这个游戏画面，描述你看到了什么"):
        """使用AI模型分析图片"""
        try:
            print("正在调用AI模型分析图片...")
            
            # 构造消息
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
            
            # 调用API
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            # 提取回复内容
            response = completion.choices[0].message.content
            print(response)
            return response
            
        except Exception as e:
            print(f"AI分析失败: {e}")
            return None
    
    def run(self, custom_prompt=None):
        """执行完整流程：捕获->截取->分析"""
        print("=" * 60)
        print("Magicraft游戏画面AI分析工具")
        print("=" * 60)
        
        # 1. 捕获游戏画面
        print("\n[步骤1] 捕获Magicraft游戏画面...")
        frame = self.capture_magicraft_screen()
        if frame is None:
            return
        
        # 保存原始截图用于查看
        cv2.imwrite(str(CAPTURE_DIR / "magicraft_full_screen.jpg"), frame)
        print("✓ 已保存完整画面到: magicraft_full_screen.jpg")
        
        # 2. 截取左上角区域
        print("\n[步骤2] 截取左上角300x1000区域...")
        cropped = self.crop_region(frame)
        if cropped is None:
            return
        time.sleep(2)
        # 保存截取的区域
        cv2.imwrite(str(CAPTURE_DIR / "magicraft_cropped.jpg"), cropped)
        print("✓ 已保存截取区域到: magicraft_cropped.jpg")
        
        # 3. 转换为base64
        print("\n[步骤3] 转换为base64编码...")
        image_base64 = self.image_to_base64(cropped)
        print("✓ 编码完成")
        
        # 4. AI分析
        print("\n[步骤4] 发送给AI模型分析...")
        prompt = custom_prompt or "请分析这个游戏画面，第一行十个法术槽有几个法术？。不要使用markdown语法。以json格式返回"
        result = self.analyze_image(image_base64, prompt)
        
        if result:
            print("\n" + "=" * 60)
            print("AI分析结果:")
            print("=" * 60)
            print(result)
            print("=" * 60)
            
            # 保存分析结果
            with open(OUTPUT_DIR / "analysis_result.txt", "w", encoding="utf-8") as f:
                f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"提示词: {prompt}\n")
                f.write("-" * 60 + "\n")
                f.write(result)
            print("\n✓ 分析结果已保存到: analysis_result.txt")
        else:
            print("✗ AI分析失败")


def main():
    """主函数"""
    analyzer = GameAnalyzer()
    
    # 可以使用自定义提示词
    # custom_prompt = "这个画面中有哪些游戏元素？包括血量、法力、技能、物品等信息"
    
    analyzer.run()


if __name__ == "__main__":
    main()
