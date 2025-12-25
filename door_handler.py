"""门处理逻辑模块"""
import time
import threading
import pydirectinput


class DoorHandler:
    """门处理类"""
    
    def __init__(self, logger):
        self.logger = logger
        self.near_door = False
        self.door_enter_time = 0
    
    def handle_door_sequence(self, mc, door, controller):
        """处理门序列：分析法术 -> 按E进门"""
        try:
            self.logger.log("开始门处理序列")
            
            # 先等待一下，确保游戏状态稳定
            time.sleep(1.0)
            
            # 执行法术分析
            self.logger.log("执行法术分析...")
            self.execute_spell_analysis()
            
            # 等待并按E进门
            time.sleep(1.0)
            self.logger.log("按E键进入门")
            pydirectinput.press('e')
            time.sleep(0.5)
            
            # 记录进门时间
            self.door_enter_time = time.time()
            
            # 重置状态
            self.near_door = False
            
            # 恢复自动控制（但要在2秒后才生效）
            controller.auto_aim_enabled = True
            controller.auto_dodge_enabled = True
            controller.auto_attack_enabled = True
            
            self.logger.log("门处理序列完成，等待2秒后恢复自动控制")
            
        except Exception as e:
            self.logger.log(f"门处理序列失败: {e}", 'error')
            import traceback
            self.logger.logger.error(traceback.format_exc())
            # 即使失败也重置状态并恢复控制
            self.near_door = False
            self.door_enter_time = time.time()
            controller.auto_aim_enabled = True
            controller.auto_dodge_enabled = True
            controller.auto_attack_enabled = True
    
    def execute_spell_analysis(self):
        """执行法术分析流程"""
        try:
            self.logger.log("导入分析模块...", 'info')
            from analyze_spells import SpellAnalyzer
            
            # 创建分析器实例
            analyzer = SpellAnalyzer()
            
            # 运行分析流程
            analyzer.run()
            
            self.logger.log("法术分析完成！", 'info')
        except Exception as e:
            self.logger.log(f"执行失败: {e}", 'error')
            import traceback
            self.logger.logger.error(traceback.format_exc())

