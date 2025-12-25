"""
游戏选项选择脚本
功能：截取游戏窗口 -> OCR识别选项 -> 文字模型选择
"""

import json
import re
import time
import base64
from typing import Dict, List, Optional, Any

import cv2

from get_game_window import GameWindowCapture
from omni_models.omni import (
    get_multimodal_client,
    get_multimodal_model,
    get_text_client,
    get_text_model
)


class OptionSelector:
    """游戏选项选择器"""
    
    def __init__(self):
        self.capturer = GameWindowCapture()
        self.multimodal_client = get_multimodal_client()
        self.multimodal_model = get_multimodal_model()
        self.text_client = get_text_client()
        self.text_model = get_text_model()
        
    def capture_game_window(self) -> Optional[Any]:
        """截取游戏窗口图片"""
        print("[步骤1] 截取游戏窗口...")
        
        if not self.capturer.select_magicraft_window():
            print("✗ 无法找到Magicraft游戏窗口")
            return None
        
        frame = self.capturer.capture_frame()
        if frame is None:
            print("✗ 截取窗口失败")
            return None
        
        print("✓ 成功截取游戏窗口")
        return frame
    
    def image_to_base64(self, image) -> str:
        """将OpenCV图像转换为base64编码"""
        # 将BGR转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 编码为JPEG
        _, buffer = cv2.imencode('.jpg', image_rgb)
        
        # 转换为base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    
    def ocr_options(self, image_base64: str) -> Optional[List[Dict[str, Any]]]:
        """使用多模态模型进行OCR，识别界面上的选项"""
        print("[步骤2] 使用多模态模型进行OCR识别选项...")
        
        prompt = """请仔细分析这个游戏画面，识别界面上显示的所有选项（通常有3-5个选项）。

请以JSON格式返回，格式如下：
{
  "options": [
    {
      "index": 1,
      "text": "选项1的完整文字内容",
      "description": "选项1的详细描述（如果有）"
    },
    {
      "index": 2,
      "text": "选项2的完整文字内容",
      "description": "选项2的详细描述（如果有）"
    }
  ]
}

要求：
1. 必须识别出所有可见的选项（通常3-5个）
2. 每个选项的text字段要包含该选项的完整文字内容
3. 如果选项有描述或说明文字，也包含在description字段中
4. 只返回JSON格式，不要其他文字
5. 如果没有识别到选项，返回空数组：{"options": []}"""
        
        try:
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
            
            completion = self.multimodal_client.chat.completions.create(
                model=self.multimodal_model,
                messages=messages
            )
            
            response = completion.choices[0].message.content
            print(f"多模态模型回复:\n{response}")
            
            # 解析JSON
            options_data = self.parse_json(response)
            if options_data and isinstance(options_data, dict):
                options = options_data.get("options", [])
                if options:
                    print(f"✓ 成功识别 {len(options)} 个选项")
                    for opt in options:
                        print(f"  选项{opt.get('index', '?')}: {opt.get('text', '')[:50]}...")
                    return options
                else:
                    print("⚠ 未识别到任何选项")
                    return []
            else:
                print("✗ 无法解析OCR结果")
                return None
                
        except Exception as e:
            print(f"✗ OCR识别失败: {e}")
            return None
    
    def parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """从文本中提取并解析JSON"""
        if not text:
            return None
        
        try:
            return json.loads(text)
        except:
            pass
        
        # 尝试提取JSON部分（可能在markdown代码块中）
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL)
        if matches:
            try:
                return json.loads(matches[0])
            except:
                pass
        
        # 尝试提取花括号中的内容
        json_pattern = r'(\{.*?\})'
        matches = re.findall(json_pattern, text, re.DOTALL)
        if matches:
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue
        
        return None
    
    def build_selection_prompt(self, options: List[Dict[str, Any]], context: str = "") -> str:
        """构建选择提示词"""
        options_text = []
        for opt in options:
            idx = opt.get('index', 0)
            text = opt.get('text', '')
            desc = opt.get('description', '')
            
            option_line = f"选项{idx}: {text}"
            if desc:
                option_line += f"\n  描述: {desc}"
            options_text.append(option_line)
        
        options_summary = "\n".join(options_text)
        
        prompt = f"""以下是游戏界面上的选项，请根据上下文选择最合适的选项。

{context if context else "请分析这些选项，选择最合适的一个。"}

选项列表：
{options_summary}

请只返回JSON格式：
{{
  "selected_index": 1,
  "reason": "选择理由"
}}

selected_index是选项的index值（从1开始），reason用一句话说明选择理由。"""
        
        return prompt
    
    def select_option(self, options: List[Dict[str, Any]], context: str = "") -> Optional[Dict[str, Any]]:
        """使用文字模型选择选项"""
        print("[步骤3] 使用文字模型进行选择...")
        
        if not options:
            print("✗ 没有可选项")
            return None
        
        selection_prompt = self.build_selection_prompt(options, context)
        
        try:
            completion = self.text_client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": selection_prompt}]
                    }
                ]
            )
            
            response = completion.choices[0].message.content
            print(f"文字模型回复:\n{response}")
            
            selection = self.parse_json(response)
            if selection and isinstance(selection, dict):
                selected_idx = selection.get("selected_index")
                reason = selection.get("reason", "")
                
                # 验证选择的索引是否有效
                valid_indices = [opt.get('index') for opt in options]
                if selected_idx in valid_indices:
                    selected_option = next(
                        (opt for opt in options if opt.get('index') == selected_idx),
                        None
                    )
                    if selected_option:
                        print(f"✓ 选择结果: 选项{selected_idx} - {selected_option.get('text', '')[:50]}...")
                        print(f"  理由: {reason}")
                        return {
                            "selected_index": selected_idx,
                            "selected_text": selected_option.get('text', ''),
                            "reason": reason
                        }
                else:
                    print(f"⚠ 选择的索引 {selected_idx} 不在有效范围内 {valid_indices}")
                    return None
            else:
                print("✗ 无法解析选择结果")
                return None
                
        except Exception as e:
            print(f"✗ 文字模型选择失败: {e}")
            return None
    
    def run(self, context: str = ""):
        """执行完整流程"""
        print("=" * 60)
        print("游戏选项选择工具")
        print("=" * 60)
        
        # 1. 截取游戏窗口
        frame = self.capture_game_window()
        if frame is None:
            return None
        
        # 2. 转换为base64
        image_base64 = self.image_to_base64(frame)
        
        # 3. OCR识别选项
        options = self.ocr_options(image_base64)
        if options is None:
            print("✗ OCR识别失败")
            return None
        
        if not options:
            print("⚠ 未识别到任何选项")
            return None
        
        # 4. 文字模型选择
        selection = self.select_option(options, context)
        
        if selection:
            print("\n" + "=" * 60)
            print("最终选择结果:")
            print("=" * 60)
            print(json.dumps(selection, ensure_ascii=False, indent=2))
            
            # 保存结果
            result = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "options": options,
                "selection": selection
            }
            
            result_file = f"option_selection_result_{int(time.time())}.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n✓ 结果已保存到: {result_file}")
            
            return selection
        else:
            print("✗ 选择失败")
            return None


def main():
    """主函数"""
    selector = OptionSelector()
    
    # 可以传入上下文信息，帮助模型做出更好的选择
    context = ""  # 例如: "当前目标是提升战斗力，优先选择能增强攻击的选项"
    
    selector.run(context=context)


if __name__ == "__main__":
    main()

