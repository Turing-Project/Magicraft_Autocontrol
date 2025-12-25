import cv2
import numpy as np
import win32gui
import win32ui
import win32con
import win32api
from ctypes import windll
from PIL import Image
import time
import threading
from typing import Callable, Optional, Tuple, List
import queue


class GameWindowCapture:
    """游戏窗口画面捕获类，支持多种捕获方法"""
    
    def __init__(self):
        self.hwnd = None
        self.capture_method = "win32api"  # 默认使用win32api方法
        self.is_capturing = False
        self.frame_queue = queue.Queue(maxsize=10)  # 限制队列大小避免内存溢出
        self.capture_thread = None
        self.frame_callbacks = []  # 帧处理回调函数列表
        
    def find_game_windows(self) -> List[Tuple[str, int]]:
        """查找所有游戏窗口"""
        windows = []
        
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd)
                if window_title and len(window_title) > 0:
                    # 过滤掉系统窗口
                    if not any(skip in window_title.lower() for skip in 
                             ['program manager', 'desktop', 'taskbar', 'start menu']):
                        windows.append((window_title, hwnd))
            return True
        
        win32gui.EnumWindows(enum_windows_callback, windows)
        return windows
    
    def select_window(self, window_title: str = None, hwnd: int = None) -> bool:
        """选择要捕获的窗口"""
        if hwnd:
            self.hwnd = hwnd
        elif window_title:
            windows = self.find_game_windows()
            for title, hwnd in windows:
                if window_title.lower() in title.lower():
                    self.hwnd = hwnd
                    break
        else:
            # 如果没有指定，选择第一个可见窗口
            windows = self.find_game_windows()
            if windows:
                self.hwnd = windows[0][1]
        
        if self.hwnd and win32gui.IsWindow(self.hwnd):
            return True
        return False
    
    def get_window_info(self) -> dict:
        """获取窗口信息"""
        if not self.hwnd:
            return {}
        
        try:
            rect = win32gui.GetWindowRect(self.hwnd)
            title = win32gui.GetWindowText(self.hwnd)
            return {
                'title': title,
                'hwnd': self.hwnd,
                'rect': rect,
                'left': rect[0],
                'top': rect[1],
                'width': rect[2] - rect[0],
                'height': rect[3] - rect[1]
            }
        except:
            return {}
    
    def capture_frame_win32api(self) -> Optional[np.ndarray]:
        """使用Win32 API捕获窗口画面"""
        if not self.hwnd:
            return None
        
        try:
            # 获取窗口设备上下文
            hwndDC = win32gui.GetWindowDC(self.hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            
            # 获取窗口尺寸
            rect = win32gui.GetWindowRect(self.hwnd)
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            
            # 创建位图
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
            saveDC.SelectObject(saveBitMap)
            
            # 捕获画面
            result = windll.user32.PrintWindow(self.hwnd, saveDC.GetSafeHdc(), 3)
            
            if result == 1:
                # 获取位图数据
                bmpinfo = saveBitMap.GetInfo()
                bmpstr = saveBitMap.GetBitmapBits(True)
                
                # 转换为numpy数组
                img = np.frombuffer(bmpstr, dtype='uint8')
                img.shape = (height, width, 4)  # BGRA格式
                
                # 转换为BGR格式（OpenCV标准）
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                # 清理资源
                win32gui.DeleteObject(saveBitMap.GetHandle())
                saveDC.DeleteDC()
                mfcDC.DeleteDC()
                win32gui.ReleaseDC(self.hwnd, hwndDC)
                
                return img
            
            # 清理资源
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, hwndDC)
            
        except Exception as e:
            print(f"Win32API捕获失败: {e}")
        
        return None
    
    def capture_frame_pil(self) -> Optional[np.ndarray]:
        """使用PIL捕获窗口画面（备用方法）"""
        if not self.hwnd:
            return None
        
        try:
            rect = win32gui.GetWindowRect(self.hwnd)
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            
            # 使用PIL的ImageGrab
            from PIL import ImageGrab
            bbox = (rect[0], rect[1], rect[2], rect[3])
            pil_image = ImageGrab.grab(bbox)
            
            # 转换为OpenCV格式
            img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return img
            
        except Exception as e:
            print(f"PIL捕获失败: {e}")
        
        return None
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """捕获当前帧"""
        if self.capture_method == "win32api":
            return self.capture_frame_win32api()
        elif self.capture_method == "pil":
            return self.capture_frame_pil()
        else:
            return self.capture_frame_win32api()
    
    def add_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """添加帧处理回调函数"""
        self.frame_callbacks.append(callback)
    
    def remove_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """移除帧处理回调函数"""
        if callback in self.frame_callbacks:
            self.frame_callbacks.remove(callback)
    
    def _capture_loop(self, fps: int = 30):
        """捕获循环"""
        frame_time = 1.0 / fps
        frame_count = 0
        failed_count = 0
        
        print(f"开始捕获循环，目标FPS: {fps}")
        
        while self.is_capturing:
            start_time = time.time()
            
            # 捕获帧
            frame = self.capture_frame()
            
            if frame is not None:
                frame_count += 1
                failed_count = 0  # 重置失败计数
                
                # 添加到队列
                try:
                    if not self.frame_queue.full():
                        self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # 队列满了，移除最旧的帧
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
                
                # 调用回调函数
                for callback in self.frame_callbacks:
                    try:
                        callback(frame)
                    except Exception as e:
                        print(f"回调函数执行失败: {e}")
            else:
                failed_count += 1
                if failed_count % 30 == 0:  # 每30次失败输出一次
                    print(f"捕获失败 {failed_count} 次，窗口可能未正确选择")
            
            # 控制帧率
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def start_capture(self, fps: int = 30):
        """开始捕获"""
        if self.is_capturing:
            print("已经在捕获中")
            return
        
        if not self.hwnd:
            print("请先选择窗口")
            return
        
        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop, args=(fps,))
        self.capture_thread.daemon = True
        self.capture_thread.start()
        print(f"开始捕获窗口画面，帧率: {fps} FPS")
    
    def stop_capture(self):
        """停止捕获"""
        if not self.is_capturing:
            print("没有在捕获")
            return
        
        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join()
        print("停止捕获")
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """获取最新的一帧"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_all_frames(self) -> List[np.ndarray]:
        """获取队列中的所有帧"""
        frames = []
        while not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get_nowait()
                frames.append(frame)
            except queue.Empty:
                break
        return frames
    
    def set_capture_method(self, method: str):
        """设置捕获方法"""
        if method in ["win32api", "pil"]:
            self.capture_method = method
            print(f"捕获方法设置为: {method}")
        else:
            print("不支持的捕获方法，支持: win32api, pil")
    
    def select_magicraft_window(self) -> bool:
        """专门选择Magicraft游戏窗口"""
        windows = self.find_game_windows()
        
        
        
        # 查找包含Magicraft的窗口
        for title, hwnd in windows:
            if "magicraft" in title.lower():
                print(f"找到Magicraft窗口: {title}")
                if self.select_window(hwnd=hwnd):
                    # 验证窗口是否真的被选中
                    if self.hwnd and win32gui.IsWindow(self.hwnd):
                        print(f"✓ 成功选择窗口: {title}")
                        return True
                    else:
                        print(f"✗ 窗口选择失败: {title}")
                        return False
                else:
                    print(f"✗ 无法选择窗口: {title}")
                    return False
        
        print("未找到Magicraft游戏窗口")
        return False


# 示例使用代码
def example_frame_processor(frame: np.ndarray):
    """示例帧处理函数"""
    # 这里可以添加您的图像处理逻辑
    # 例如：目标检测、图像分析等
    
    # 显示帧信息
    height, width = frame.shape[:2]
    print(f"处理帧: {width}x{height}, 时间: {time.time()}")
    
    # 可以保存帧
    # cv2.imwrite(f"frame_{int(time.time()*1000)}.jpg", frame)


def main():
    """主函数示例 - 专门用于Magicraft游戏"""
    # 创建捕获器
    capturer = GameWindowCapture()
    
    # 直接选择Magicraft窗口
    if capturer.select_magicraft_window():
        window_info = capturer.get_window_info()
        print(f"成功选择窗口: {window_info['title']}")
        print(f"窗口尺寸: {window_info['width']}x{window_info['height']}")
        
        # 添加帧处理回调
        capturer.add_frame_callback(example_frame_processor)
        
        # 开始捕获
        capturer.start_capture(fps=30)
        
        try:
            print("开始捕获Magicraft游戏画面，按Ctrl+C停止...")
            # 持续运行直到用户中断
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("用户中断")
        finally:
            # 停止捕获
            capturer.stop_capture()
    else:
        print("无法找到或选择Magicraft游戏窗口")
        print("请确保:")
        print("1. Magicraft游戏正在运行")
        print("2. 游戏窗口标题包含'Magicraft'")
        print("3. 以管理员身份运行此脚本")


if __name__ == "__main__":
    main()
