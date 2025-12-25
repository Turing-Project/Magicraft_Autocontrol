"""自动攻击逻辑模块"""
import numpy as np


class AutoAttack:
    """自动攻击类"""
    
    def __init__(self, control_thread, window_offset_x, window_offset_y):
        self.control_thread = control_thread
        self.window_offset_x = window_offset_x
        self.window_offset_y = window_offset_y
    
    def set_frame_count(self, frame_count):
        """设置帧计数（由控制器同步）"""
        self.frame_count = frame_count
    
    def execute(self, monsters, mc=None):
        """自动攻击逻辑 - 优先攻击离主角最近的怪物"""
        
        # 如果有主角位置，选择最近的怪物；否则选择置信度最高的
        if mc:
            mc_center = mc['center']
            
            # 计算每个怪物到主角的距离
            monsters_with_distance = []
            for monster in monsters:
                monster_center = monster['center']
                dx = monster_center[0] - mc_center[0]
                dy = monster_center[1] - mc_center[1]
                distance = np.sqrt(dx**2 + dy**2)
                monsters_with_distance.append({
                    'monster': monster,
                    'distance': distance
                })
            
            # 选择最近的怪物
            closest = min(monsters_with_distance, key=lambda m: m['distance'])
            best_monster = closest['monster']
            distance_to_mc = closest['distance']
        else:
            # 没有主角位置时，选择置信度最高的
            best_monster = max(monsters, key=lambda m: m['confidence'])
            distance_to_mc = 0
        
        target_center = best_monster['center']
        confidence = best_monster['confidence']
        bbox = best_monster['bbox']  # (x1, y1, x2, y2)
        
        # 计算瞄准点：框的中心偏下一点（约框高度的20%）
        box_height = bbox[3] - bbox[1]
        aim_offset_y = int(box_height * 0.2)  # 向下偏移20%
        
        # 计算全局坐标（加上窗口偏移和瞄准偏移）
        global_x = target_center[0] + self.window_offset_x
        global_y = target_center[1] + self.window_offset_y + aim_offset_y  # 向下偏移
        
        # 每帧都更新鼠标位置（持续跟踪）
        # 限制队列长度，避免堆积
        if len(self.control_thread.control_queue) < 10:
            # 添加最新的鼠标位置
            self.control_thread.move_mouse(global_x, global_y)
            
            # 添加按K键指令
            self.control_thread.press_k()
        
        # 每60帧输出一次日志
        if self.frame_count % 60 == 0:
            if mc:
                print(f"[指令] 锁定最近怪物 距离:{distance_to_mc:.0f}px 坐标:({global_x},{global_y}) 队列:{len(self.control_thread.control_queue)}")
            else:
                print(f"[指令] 攻击怪物 坐标:({global_x},{global_y}) 队列:{len(self.control_thread.control_queue)}")


def main():
    """测试主函数"""
    print("=" * 60)
    print("=== AutoAttack 模块测试 ===")
    print("=" * 60)
    
    # 创建模拟的控制线程（不启动实际线程，只用于测试）
    class MockControlThread:
        def __init__(self):
            self.control_queue = []
        
        def move_mouse(self, x, y):
            self.control_queue.append({'type': 'move_mouse', 'x': x, 'y': y})
            print(f"[测试] 添加鼠标移动指令: ({x}, {y})")
        
        def press_k(self):
            self.control_queue.append({'type': 'press_k'})
            print(f"[测试] 添加按K键指令")
    
    # 创建 AutoAttack 实例
    control_thread = MockControlThread()
    window_offset_x = 100  # 模拟窗口偏移
    window_offset_y = 50
    
    auto_attack = AutoAttack(control_thread, window_offset_x, window_offset_y)
    
    # 测试场景1: 有主角位置，多个怪物
    print("\n[测试场景1] 有主角位置，选择最近的怪物")
    print("-" * 60)
    
    mc = {
        'center': (400, 300),  # 主角在屏幕中心
        'bbox': (350, 250, 450, 350)
    }
    
    monsters = [
        {
            'class_name': 'monster',
            'confidence': 0.85,
            'center': (500, 300),  # 距离主角100px（右侧）
            'bbox': (450, 250, 550, 350)
        },
        {
            'class_name': 'monster',
            'confidence': 0.90,
            'center': (400, 200),  # 距离主角100px（上方）
            'bbox': (350, 150, 450, 250)
        },
        {
            'class_name': 'monster',
            'confidence': 0.75,
            'center': (450, 350),  # 距离主角约70px（右下）
            'bbox': (400, 300, 500, 400)
        }
    ]
    
    auto_attack.set_frame_count(60)  # 设置为60帧，触发日志输出
    auto_attack.execute(monsters, mc)
    
    print(f"\n[结果] 队列中有 {len(control_thread.control_queue)} 个指令")
    print(f"[结果] 指令详情:")
    for i, cmd in enumerate(control_thread.control_queue[:5], 1):  # 只显示前5个
        print(f"  {i}. {cmd}")
    
    # 清空队列
    control_thread.control_queue.clear()
    
    # 测试场景2: 没有主角位置，选择置信度最高的怪物
    print("\n[测试场景2] 没有主角位置，选择置信度最高的怪物")
    print("-" * 60)
    
    monsters_2 = [
        {
            'class_name': 'monster',
            'confidence': 0.70,
            'center': (300, 200),
            'bbox': (250, 150, 350, 250)
        },
        {
            'class_name': 'monster',
            'confidence': 0.95,  # 置信度最高
            'center': (600, 400),
            'bbox': (550, 350, 650, 450)
        },
        {
            'class_name': 'monster',
            'confidence': 0.80,
            'center': (500, 500),
            'bbox': (450, 450, 550, 550)
        }
    ]
    
    auto_attack.set_frame_count(60)
    auto_attack.execute(monsters_2, mc=None)
    
    print(f"\n[结果] 队列中有 {len(control_thread.control_queue)} 个指令")
    print(f"[结果] 指令详情:")
    for i, cmd in enumerate(control_thread.control_queue[:5], 1):
        print(f"  {i}. {cmd}")
    
    # 清空队列
    control_thread.control_queue.clear()
    
    # 测试场景3: 队列限制测试
    print("\n[测试场景3] 队列限制测试（限制为10）")
    print("-" * 60)
    
    # 快速执行多次，测试队列限制
    for i in range(15):
        auto_attack.set_frame_count(i)
        auto_attack.execute(monsters, mc)
    
    print(f"\n[结果] 队列中有 {len(control_thread.control_queue)} 个指令（应该不超过10）")
    
    # 测试场景4: 瞄准点计算测试
    print("\n[测试场景4] 瞄准点计算测试")
    print("-" * 60)
    
    monster_test = {
        'class_name': 'monster',
        'confidence': 0.85,
        'center': (500, 300),
        'bbox': (450, 250, 550, 350)  # 高度100px
    }
    
    # 计算瞄准点
    bbox = monster_test['bbox']
    target_center = monster_test['center']
    box_height = bbox[3] - bbox[1]
    aim_offset_y = int(box_height * 0.2)
    global_x = target_center[0] + window_offset_x
    global_y = target_center[1] + window_offset_y + aim_offset_y
    
    print(f"[信息] 怪物边界框: {bbox}")
    print(f"[信息] 怪物中心: {target_center}")
    print(f"[信息] 框高度: {box_height}px")
    print(f"[信息] 瞄准偏移: {aim_offset_y}px (向下偏移20%)")
    print(f"[信息] 窗口偏移: ({window_offset_x}, {window_offset_y})")
    print(f"[信息] 最终瞄准坐标: ({global_x}, {global_y})")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
