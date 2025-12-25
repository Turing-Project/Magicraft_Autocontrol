"""控制线程模块 - 处理鼠标和键盘输入"""
import time
import threading
import pyautogui
import pydirectinput


class ControlThread:
    """控制线程类 - 高速处理鼠标和键盘指令"""
    
    def __init__(self):
        self.control_queue = []
        self.control_thread = None
        self.control_running = False
        
        # 配置pydirectinput（游戏专用）
        pydirectinput.PAUSE = 0.01
        pyautogui.PAUSE = 0.01
        pyautogui.FAILSAFE = False
    
    def start(self):
        """启动控制线程"""
        self.control_running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        print("[控制线程] 已启动独立的鼠标和键盘控制线程")
    
    def stop(self):
        """停止控制线程"""
        self.control_running = False
        if self.control_thread:
            self.control_thread.join(timeout=1.0)
        print("[控制线程] 已停止")
    
    def _control_loop(self):
        """控制线程主循环 - 高速处理模式"""
        print("[控制线程] 开始运行...")
        
        exec_count = 0
        
        while self.control_running:
            try:
                if self.control_queue:
                    # 取出并执行控制指令
                    cmd = self.control_queue.pop(0)
                    cmd_type = cmd.get('type')
                    
                    if cmd_type == 'move_mouse':
                        x, y = cmd['x'], cmd['y']
                        pyautogui.moveTo(x, y)
                        exec_count += 1
                    
                    elif cmd_type == 'press_key':
                        key = cmd['key']
                        duration = cmd.get('duration', 0.15)
                        pydirectinput.keyDown(key)
                        time.sleep(duration)
                        pydirectinput.keyUp(key)
                        exec_count += 1
                    
                    elif cmd_type == 'press_k':
                        pydirectinput.press('k')
                        exec_count += 1
                    
                    # 每100个指令输出一次统计
                    if exec_count % 100 == 0:
                        print(f"[控制线程] 已执行 {exec_count} 个指令，队列:{len(self.control_queue)}")
                
                time.sleep(0.005)  # 减少延迟，提高响应速度
                
            except Exception as e:
                print(f"[控制线程错误] {e}")
                time.sleep(0.1)
    
    def add_command(self, cmd_type, **kwargs):
        """添加控制指令到队列"""
        cmd = {'type': cmd_type, **kwargs}
        self.control_queue.append(cmd)
    
    def move_mouse(self, x, y):
        """添加鼠标移动指令"""
        self.add_command('move_mouse', x=x, y=y)
    
    def press_key(self, key, duration=0.15):
        """添加按键指令"""
        self.add_command('press_key', key=key, duration=duration)
    
    def press_k(self):
        """添加按K键指令"""
        self.add_command('press_k')

