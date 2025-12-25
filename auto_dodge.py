"""自动躲避逻辑模块"""
import time
import numpy as np


class AutoDodge:
    """自动躲避类"""
    
    def __init__(self, control_thread, logger):
        self.control_thread = control_thread
        self.logger = logger
        self.last_action_time = time.time()
        self.action_cooldown = 0.3  # 动作冷却时间（秒）
    
    def set_frame_count(self, frame_count):
        """设置帧计数（由控制器同步）"""
        self.frame_count = frame_count
    
    def execute(self, mc, monsters, bullets, frame_shape):
        """执行自动躲避逻辑"""
        current_time = time.time()
        if current_time - self.last_action_time < self.action_cooldown:
            return
        mc_center = mc['center']
        height, width = frame_shape[:2]
        
        # 计算所有威胁的方向和距离
        threats = []
        threat_count = 0
        
        for monster in monsters:
            monster_center = monster['center']
            dx = monster_center[0] - mc_center[0]
            dy = monster_center[1] - mc_center[1]
            distance = np.sqrt(dx**2 + dy**2)
            if distance < 400:  # 400像素内的怪物视为威胁
                threats.append({'dx': dx, 'dy': dy, 'distance': distance, 'weight': 2.0, 'type': 'monster'})
                threat_count += 1
        
        for bullet in bullets:
            bullet_center = bullet['center']
            dx = bullet_center[0] - mc_center[0]
            dy = bullet_center[1] - mc_center[1]
            distance = np.sqrt(dx**2 + dy**2)
            if distance < 300:  # 300像素内的子弹视为威胁
                threats.append({'dx': dx, 'dy': dy, 'distance': distance, 'weight': 3.0, 'type': 'bullet'})
                threat_count += 1
        
        if not threats:
            return
        
        # 计算最佳逃跑方向（远离威胁）
        escape_x, escape_y = 0, 0
        for threat in threats:
            # 距离越近，权重越大
            weight = threat['weight'] * (1.0 / max(threat['distance'], 50))
            escape_x -= threat['dx'] * weight
            escape_y -= threat['dy'] * weight
        
        # 决定按键
        self.last_action_time = current_time
        
        # 根据逃跑方向按WASD
        action = ""
        key_to_press = None
        
        if abs(escape_x) > abs(escape_y):
            if escape_x > 0:
                key_to_press = 'd'
                action = "D(右)"
            else:
                key_to_press = 'a'
                action = "A(左)"
        else:
            if escape_y > 0:
                key_to_press = 's'
                action = "S(下)"
            else:
                key_to_press = 'w'
                action = "W(上)"
        
        # 执行按键（长按0.15秒）
        if key_to_press:
            # 减少日志输出
            if self.frame_count % 30 == 0:
                self.logger.log(f"{threat_count}个威胁 → {action}", 'info')
            self.control_thread.press_key(key_to_press, duration=0.15)

