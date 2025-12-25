"""
商店购买决策脚本
功能：分析商店商品，判断应该购买哪些商品
"""

import json
import re
import time
import sys
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
import pyautogui
import pydirectinput
import requests
import torch
from capture_and_analyze import GameAnalyzer
from omni_models.omni import get_text_client, get_text_model
from get_game_window import GameWindowCapture
from utils.paths import DATA_DIR, OUTPUT_DIR, CAPTURE_DIR


class StorePurchaseDecision:
    """商店购买决策器"""
    
    def __init__(self):
        self.game_analyzer = GameAnalyzer()
        # self.door_analyzer = DoorSceneAnalyzer()  # 已删除，如需使用请重新实现
        self.text_client = get_text_client()
        self.text_model = get_text_model()
        
        # 商店API地址（从环境变量读取）
        import os
        from dotenv import load_dotenv
        from pathlib import Path
        env_path = Path(__file__).parent / '.env'
        load_dotenv(dotenv_path=env_path)
        self.store_api_url = os.getenv('STORE_ENDPOINT', 'http://localhost:1234/store')
        
        # 货架坐标（7个位置：第一行4个，第二行3个）
        self.shelf_coordinates = []
        self.load_shelf_coordinates()
        
        # 商品缓存（法杖和法术分开存储）
        self.wand_cache_file = "store_wand_cache.json"  # 法杖缓存
        self.spell_cache_file = "store_spell_cache.json"  # 法术缓存
        self.wand_cache = {}
        self.spell_cache = {}
        self.load_item_cache()
        
        # 配置pyautogui和pydirectinput
        pyautogui.PAUSE = 0.1
        pyautogui.FAILSAFE = False
        pydirectinput.PAUSE = 0.01
        
        # YOLOv5模型用于检测角色位置
        self.detection_model = None
        base_dir = Path(__file__).parent
        # 与 realtime_detect.py 使用同一套权重与 data.yaml
        self.detection_weights_path = base_dir / "data1" / "yolov5" / "runs" / "train" / "exp8" / "weights" / "best.pt"
        self.detection_data_path = base_dir / "data1" / "data.yaml"
        self.load_detection_model()
        # 记录最近一次检测到的角色位置与时间，作为短期回退
        self._last_mc_center: Optional[Tuple[int, int]] = None
        self._last_mc_time: float = 0.0
        self._mc_miss_count: int = 0
        
        # 游戏窗口捕获器
        self.capturer = GameWindowCapture()
        self.capturer.select_magicraft_window()
        # 已拥有物品概览
        self.owned_items_summary = {}
    
    def load_shelf_coordinates(self, filename="mouse_coordinates1.json"):
        """加载货架坐标：直接使用旧导航脚本中的7个客户端坐标"""
        # 目标点（客户端坐标，单位像素）来自 old/navigate_mc_to_points.py L83-L89
        self.shelf_coordinates = [
            (879, 710),
            (1142, 709),
            (1425, 709),
            (1652, 710),
            (1019, 930),
            (1272, 932),
            (1548, 932),
        ]
        print(f"✓ 使用旧导航坐标，已加载 {len(self.shelf_coordinates)} 个货架坐标:")
        for i, (x, y) in enumerate(self.shelf_coordinates, 1):
            print(f"  货架 {i}: ({x}, {y})")
    
    def load_item_cache(self):
        """加载商品分析缓存（法杖和法术分开加载）"""
        # 先尝试迁移旧缓存（如果存在）
        self._migrate_old_cache()
        
        # 加载法杖缓存
        try:
            with open(self.wand_cache_file, 'r', encoding='utf-8') as f:
                self.wand_cache = json.load(f)
                wand_count = len([k for k in self.wand_cache.keys() if k.startswith('id_')])
                print(f"✓ 已加载法杖缓存: {wand_count} 个法杖记录")
        except FileNotFoundError:
            self.wand_cache = {}
            print(f"✓ 法杖缓存文件不存在，将创建新文件: {self.wand_cache_file}")
        except Exception as e:
            print(f"⚠ 加载法杖缓存失败: {e}")
            self.wand_cache = {}
        
        # 加载法术缓存
        try:
            with open(self.spell_cache_file, 'r', encoding='utf-8') as f:
                self.spell_cache = json.load(f)
                spell_count = len([k for k in self.spell_cache.keys() if k.startswith('id_')])
                print(f"✓ 已加载法术缓存: {spell_count} 个法术记录")
        except FileNotFoundError:
            self.spell_cache = {}
            print(f"✓ 法术缓存文件不存在，将创建新文件: {self.spell_cache_file}")
        except Exception as e:
            print(f"⚠ 加载法术缓存失败: {e}")
            self.spell_cache = {}
    
    def _migrate_old_cache(self):
        """迁移旧的统一缓存文件到新的分离缓存"""
        old_cache_file = "store_item_cache.json"
        try:
            with open(old_cache_file, 'r', encoding='utf-8') as f:
                old_cache = json.load(f)
                
            # 初始化缓存字典（如果还没有初始化）
            if not hasattr(self, 'wand_cache') or self.wand_cache is None:
                self.wand_cache = {}
            if not hasattr(self, 'spell_cache') or self.spell_cache is None:
                self.spell_cache = {}
                
            # 检查是否有数据需要迁移
            migrated_wands = 0
            migrated_spells = 0
            
            # 需要从商店数据中获取商品类型来判断
            store_data = self.fetch_store_data()
            item_type_map = {}
            if store_data:
                for item in store_data.get('store', []):
                    item_type_map[item.get('id')] = item.get('type', -1)
            
            # 遍历旧缓存，根据类型分配到新缓存
            for key, value in old_cache.items():
                if key.startswith('id_'):
                    item_id = int(key.replace('id_', ''))
                    item_type = item_type_map.get(item_id, -1)
                    
                    if item_type == 0:  # 法杖
                        if key not in self.wand_cache:
                            self.wand_cache[key] = value
                            migrated_wands += 1
                            # 迁移名称键
                            item_name = value.get('item_name', '')
                            if item_name:
                                name_key = f"name_{item_name}"
                                if name_key in old_cache:
                                    self.wand_cache[name_key] = old_cache[name_key]
                    elif item_type == 1:  # 法术
                        if key not in self.spell_cache:
                            self.spell_cache[key] = value
                            migrated_spells += 1
                            # 迁移名称键
                            item_name = value.get('item_name', '')
                            if item_name:
                                name_key = f"name_{item_name}"
                                if name_key in old_cache:
                                    self.spell_cache[name_key] = old_cache[name_key]
            
            if migrated_wands > 0 or migrated_spells > 0:
                # 保存迁移后的缓存
                if migrated_wands > 0:
                    with open(self.wand_cache_file, 'w', encoding='utf-8') as f:
                        json.dump(self.wand_cache, f, indent=2, ensure_ascii=False)
                if migrated_spells > 0:
                    with open(self.spell_cache_file, 'w', encoding='utf-8') as f:
                        json.dump(self.spell_cache, f, indent=2, ensure_ascii=False)
                print(f"✓ 已从旧缓存迁移: {migrated_wands} 个法杖, {migrated_spells} 个法术")
        except FileNotFoundError:
            # 旧缓存文件不存在，不需要迁移
            pass
        except Exception as e:
            print(f"⚠ 迁移旧缓存失败: {e}")
    
    def save_item_cache(self, item_type: int = None):
        """保存商品分析缓存（根据类型保存到对应文件）"""
        if item_type == 0:  # 法杖
            try:
                with open(self.wand_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.wand_cache, f, indent=2, ensure_ascii=False)
                wand_count = len([k for k in self.wand_cache.keys() if k.startswith('id_')])
                print(f"  ✓ 已保存法杖缓存: {wand_count} 个法杖记录")
            except Exception as e:
                print(f"  ⚠ 保存法杖缓存失败: {e}")
        elif item_type == 1:  # 法术
            try:
                with open(self.spell_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.spell_cache, f, indent=2, ensure_ascii=False)
                spell_count = len([k for k in self.spell_cache.keys() if k.startswith('id_')])
                print(f"  ✓ 已保存法术缓存: {spell_count} 个法术记录")
            except Exception as e:
                print(f"  ⚠ 保存法术缓存失败: {e}")
        else:
            # 如果没指定类型，保存所有缓存
            try:
                with open(self.wand_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.wand_cache, f, indent=2, ensure_ascii=False)
                wand_count = len([k for k in self.wand_cache.keys() if k.startswith('id_')])
                print(f"✓ 已保存法杖缓存: {wand_count} 个法杖记录")
            except Exception as e:
                print(f"⚠ 保存法杖缓存失败: {e}")
            
            try:
                with open(self.spell_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.spell_cache, f, indent=2, ensure_ascii=False)
                spell_count = len([k for k in self.spell_cache.keys() if k.startswith('id_')])
                print(f"✓ 已保存法术缓存: {spell_count} 个法术记录")
            except Exception as e:
                print(f"⚠ 保存法术缓存失败: {e}")
    
    def get_cached_item_info(self, item_id: int, item_type: int = None):
        """从缓存中获取商品信息（根据类型从对应缓存查找）"""
        cache_key = f"id_{item_id}"
        
        # 如果指定了类型，只从对应缓存查找
        if item_type == 0:  # 法杖
            if cache_key in self.wand_cache:
                cached = self.wand_cache[cache_key]
                if isinstance(cached, dict):
                    return cached.copy()
        elif item_type == 1:  # 法术
            if cache_key in self.spell_cache:
                cached = self.spell_cache[cache_key]
                if isinstance(cached, dict):
                    return cached.copy()
        else:
            # 如果没指定类型，从两个缓存中查找
            if cache_key in self.wand_cache:
                cached = self.wand_cache[cache_key]
                if isinstance(cached, dict):
                    return cached.copy()
            if cache_key in self.spell_cache:
                cached = self.spell_cache[cache_key]
                if isinstance(cached, dict):
                    return cached.copy()
        
        return None
    
    def cache_item_info(self, item_id: int, item_name: str, item_info: Dict[str, Any], item_type: int):
        """将商品信息保存到缓存（根据类型保存到对应缓存）"""
        cache_entry = {
            'item_id': item_id,
            'item_name': item_name,
            'item_info': item_info,
            'cached_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        cache_key = f"id_{item_id}"
        
        if item_type == 0:  # 法杖
            self.wand_cache[cache_key] = cache_entry.copy()
            # 同时使用名称作为键
            if item_name:
                name_key = f"name_{item_name}"
                self.wand_cache[name_key] = f"id_{item_id}"  # 指向ID键
        elif item_type == 1:  # 法术
            self.spell_cache[cache_key] = cache_entry.copy()
            # 同时使用名称作为键
            if item_name:
                name_key = f"name_{item_name}"
                self.spell_cache[name_key] = f"id_{item_id}"  # 指向ID键
    
    def fetch_store_data(self) -> Optional[Dict[str, Any]]:
        """从接口获取商店商品数据"""
        try:
            response = requests.get(self.store_api_url, timeout=5)
            response.raise_for_status()
            data = response.json()
            print(f"✓ 已从接口获取商店数据: {len(data.get('store', []))} 个商品")
            return data
        except Exception as e:
            print(f"✗ 获取商店数据失败: {e}")
            return None
    
    def get_current_coin(self) -> int:
        """获取当前金币数量（需要重新实现，door_analyzer已删除）"""
        print("\n[获取当前金币]")
        # TODO: 重新实现获取金币的逻辑，door_analyzer已删除
        print("⚠ 获取金币功能需要重新实现")
        return 0
    
    def load_detection_model(self):
        """加载实时检测模型（与 realtime_detect.py 一致，用于定位角色）"""
        try:
            import sys
            import importlib

            # 确保 yolov5 自带的 utils 优先于项目的 utils，避免 TryExcept 冲突
            backup_utils = sys.modules.get("utils")
            if backup_utils and getattr(backup_utils, "__file__", "").endswith("train_MAGIC\\utils\\__init__.py"):
                sys.modules["project_utils_backup"] = backup_utils  # 备份以防需要
                sys.modules.pop("utils", None)

            # 动态导入 realtime_detect 以加载 RealtimeDetector
            from realtime_detect import RealtimeDetector

            weights_path = Path(self.detection_weights_path)
            data_path = Path(self.detection_data_path)
            if not weights_path.exists():
                print(f"⚠ 模型文件不存在: {weights_path}")
                return False
            if not data_path.exists():
                print(f"⚠ 数据配置文件不存在: {data_path}")
                return False
            print(f"正在加载检测模型: {weights_path}")
            device_str = "0" if torch.cuda.is_available() else ""
            if torch.cuda.is_available():
                print(f"  [GPU] 检测到GPU: {torch.cuda.get_device_name(0)}")
            else:
                print(f"  [CPU] 未检测到GPU，使用CPU模式")
            self.detection_model = RealtimeDetector(
                weights_path=str(weights_path),
                data_yaml_path=str(data_path),
                conf_thres=0.25,
                iou_thres=0.45,
                device=device_str,
                imgsz=640,
            )
            # 适当提高角色(mc)检测阈值，减少误检；其余保持默认
            try:
                self.detection_model.set_class_threshold(11, 0.30)
            except Exception:
                pass
            print("  ✓ 检测模型加载完成")
            return True
        except Exception as e:
            print(f"  ✗ 加载检测模型失败: {e}")
            return False
    
    def _get_mc_class_ids(self) -> List[int]:
        """从检测模型的类别表中找出 'mc' 对应的类别ID列表"""
        if self.detection_model is None:
            return []
        names = getattr(self.detection_model, "class_names", None)
        if not names:
            names = getattr(self.detection_model, "names", None)
        ids: List[int] = []
        if isinstance(names, dict):
            for cid, nm in names.items():
                if str(nm).lower() == "mc":
                    try:
                        ids.append(int(cid))
                    except Exception:
                        continue
        elif isinstance(names, (list, tuple)):
            for cid, nm in enumerate(names):
                if str(nm).lower() == "mc":
                    ids.append(int(cid))
        # 回退：如果没有名字为 mc 的类别，使用默认的类别ID 11（与 realtime_detect._build_objects 保持一致）
        if not ids:
            ids = [11]
        return ids
    
    def detect_character_position(self) -> Optional[Tuple[int, int]]:
        """检测角色位置，返回角色中心坐标（使用新实时检测模型）"""
        if self.detection_model is None:
            print("  ⚠ 检测模型未加载，无法检测角色位置")
            return None
        try:
            # 用窗口捕获器抓帧，保持一致
            frame = self.capturer.capture_frame()
            if frame is None:
                return None
            class_thresholds = getattr(self.detection_model, "class_thresholds", None)
            if not isinstance(class_thresholds, dict) or len(class_thresholds) == 0:
                class_thresholds = {0: 0.25}
            dets, _, _ = self.detection_model.detect_with_class_thresholds(
                frame,
                class_thresholds,
            )
            if dets is None or len(dets) == 0:
                self._mc_miss_count += 1
                # 短期回退：最近0.8秒内的坐标可复用，减少空检测导致的停滞
                now = time.time()
                if self._last_mc_center and (now - self._last_mc_time) <= 0.8:
                    if self._mc_miss_count % 5 == 0:
                        print("  ℹ 使用最近检测到的角色位置作为回退")
                    return self._last_mc_center
                if self._mc_miss_count % 5 == 0:
                    print("  ⚠ 未检测到目标")
                return None
            mc_ids = self._get_mc_class_ids()
            if not mc_ids:
                print("  ⚠ 类别表中未找到 mc 类别")
                return None
            mc_dets = [det for det in dets if int(det[5]) in mc_ids]
            if not mc_dets:
                self._mc_miss_count += 1
                now = time.time()
                if self._last_mc_center and (now - self._last_mc_time) <= 0.8:
                    if self._mc_miss_count % 5 == 0:
                        print("  ℹ 使用最近检测到的角色位置作为回退")
                    return self._last_mc_center
                if self._mc_miss_count % 5 == 0:
                    print("  ⚠ 未检测到角色")
                return None
            self._mc_miss_count = 0
            # 选取置信度最高的 mc
            best = max(mc_dets, key=lambda d: float(d[4]))
            x1, y1, x2, y2, conf, _ = best.tolist()
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            self._last_mc_center = (center_x, center_y)
            self._last_mc_time = time.time()
            print(f"  ✓ 检测到角色位置: ({center_x}, {center_y}), 置信度: {conf:.2f}")
            return center_x, center_y
        except Exception as e:
            print(f"  ✗ 检测角色位置失败: {e}")
            return None
    
    def move_character_with_wasd(self, target_x: int, target_y: int, max_steps: int = 50):
        """使用点按式WASD移动到目标位置（复用旧导航逻辑）"""
        print(f"  [移动角色] 目标位置: ({target_x}, {target_y})")
        tap_down = 0.08
        tap_gap = 0.04
        deadzone = 20
        max_duration = 20.0
        start_time = time.time()
        if getattr(self.capturer, "hwnd", None):
            ensure_focus(self.capturer.hwnd)
        while True:
            if time.time() - start_time > max_duration:
                print("  ⚠ 到达超时")
                # 释放所有按键
                pydirectinput.keyUp('w')
                pydirectinput.keyUp('a')
                pydirectinput.keyUp('s')
                pydirectinput.keyUp('d')
                return False
            current_pos = self.detect_character_position()
            if current_pos is None:
                time.sleep(0.05)
                continue
            curr_x, curr_y = current_pos
            dx = target_x - curr_x
            dy = target_y - curr_y
            if abs(dx) <= deadzone and abs(dy) <= deadzone:
                # 释放所有按键
                pydirectinput.keyUp('w')
                pydirectinput.keyUp('a')
                pydirectinput.keyUp('s')
                pydirectinput.keyUp('d')
                print("  ✓ 已到达目标区域")
                return True
            if abs(dx) > deadzone:
                if dx > 0:
                    pydirectinput.keyDown('d')
                    time.sleep(tap_down)
                    pydirectinput.keyUp('d')
                else:
                    pydirectinput.keyDown('a')
                    time.sleep(tap_down)
                    pydirectinput.keyUp('a')
                time.sleep(tap_gap)
                continue
            if abs(dy) > deadzone:
                if dy > 0:
                    pydirectinput.keyDown('s')
                    time.sleep(tap_down)
                    pydirectinput.keyUp('s')
                else:
                    pydirectinput.keyDown('w')
                    time.sleep(tap_down)
                    pydirectinput.keyUp('w')
                time.sleep(tap_gap)
                continue
    
    def move_to_shelf(self, shelf_index: int):
        """移动到指定货架位置（使用角色检测和WASD移动）"""
        if shelf_index < 0 or shelf_index >= len(self.shelf_coordinates):
            print(f"✗ 货架索引 {shelf_index} 超出范围")
            return False
        
        # 货架坐标是屏幕绝对坐标，需要转换为窗口相对坐标
        shelf_screen_x, shelf_screen_y = self.shelf_coordinates[shelf_index]
        
        # 获取窗口信息（优先客户区左上角，避免边框偏移）
        window_info = self.capturer.get_window_info()
        client_rect = None
        try:
            client_rect = self.capturer._get_client_abs_rect()
        except Exception:
            client_rect = None
        if client_rect:
            client_left = client_rect[0]
            client_top = client_rect[1]
        else:
            client_left = window_info.get('left', 0)
            client_top = window_info.get('top', 0)
        
        # 两种候选：1) 记录为屏幕坐标 -> 转客户区  2) 记录已为客户区坐标 -> 直接用
        target_from_screen_x = shelf_screen_x - client_left
        target_from_screen_y = shelf_screen_y - client_top
        target_client_x = shelf_screen_x
        target_client_y = shelf_screen_y

        # 检测当前位置，自动选择更接近的目标模式
        current_pos = self.detect_character_position()
        chosen_x, chosen_y = target_from_screen_x, target_from_screen_y
        chosen_mode = "screen->client"
        if current_pos is not None:
            curr_x, curr_y = current_pos
            d_screen = abs(target_from_screen_x - curr_x) + abs(target_from_screen_y - curr_y)
            d_client = abs(target_client_x - curr_x) + abs(target_client_y - curr_y)
            if d_client < d_screen:
                chosen_x, chosen_y = target_client_x, target_client_y
                chosen_mode = "client"
        
        print(f"  [移动到货架] 货架 {shelf_index + 1}, 屏幕坐标: ({shelf_screen_x}, {shelf_screen_y}), "
              f"客户区起点: ({client_left}, {client_top}), 模式: {chosen_mode}, 目标(客户区): ({chosen_x}, {chosen_y})")
        # 确保窗口焦点
        if getattr(self.capturer, "hwnd", None):
            ensure_focus(self.capturer.hwnd)
        
        # 使用WASD移动角色到货架位置
        success = self.move_character_with_wasd(chosen_x, chosen_y)
        
        if success:
            # 等待游戏显示商品详情窗口
            time.sleep(0.5)
            
            # 检测实际到达的角色位置（用于调试和校准）
            actual_pos = self.detect_character_position()
            if actual_pos:
                actual_x, actual_y = actual_pos
                print(f"  [调试] 实际检测到的角色位置: ({actual_x}, {actual_y})")
                print(f"  [调试] 目标位置: ({chosen_x}, {chosen_y})")
                print(f"  [调试] 位置差异: ({actual_x - chosen_x}, {actual_y - chosen_y})")
            
            print(f"  ✓ 已移动到货架 {shelf_index + 1}")
            return True
        else:
            print(f"  ✗ 移动到货架 {shelf_index + 1} 失败")
            return False
    
    def capture_full_screen(self) -> Optional[Tuple[Any, str]]:
        """截取完整游戏画面并转换为base64"""
        frame = self.game_analyzer.capture_magicraft_screen()
        if frame is None:
            return None
        
        # 保存截图用于调试
        screenshot_path = CAPTURE_DIR / f"store_item_{int(time.time())}.jpg"
        cv2.imwrite(str(screenshot_path), frame)
        print(f"  ✓ 已保存截图: {screenshot_path}")
        
        image_base64 = self.game_analyzer.image_to_base64(frame)
        return frame, image_base64
    
    def _client_to_screen(self, x: int, y: int) -> Tuple[int, int]:
        """将客户区相对坐标转换为屏幕绝对坐标"""
        client_rect = None
        try:
            client_rect = self.capturer._get_client_abs_rect()
        except Exception:
            client_rect = None
        if client_rect:
            left, top = client_rect[0], client_rect[1]
        else:
            info = self.capturer.get_window_info()
            left, top = info.get('left', 0), info.get('top', 0)
        return left + int(x), top + int(y)
    
    def analyze_owned_wands(self):
        """在购买前识别背包中已拥有的法杖（两个位置），并缓存到本地"""
        print("\n[步骤0] 识别已拥有的法杖...")
        coords_client = [(61, 118), (63, 167)]
        
        owned_wands = []
        for idx, (cx, cy) in enumerate(coords_client, start=1):
            try:
                # 聚焦窗口
                if getattr(self.capturer, "hwnd", None):
                    ensure_focus(self.capturer.hwnd)
                # 转为屏幕坐标并移动鼠标
                sx, sy = self._client_to_screen(cx, cy)
                pyautogui.moveTo(sx, sy, duration=0.2)
                time.sleep(0.4)  # 等待提示面板出现
                
                screen_result = self.capture_full_screen()
                if screen_result is None:
                    print(f"  ✗ 无法截屏（法杖 {idx}）")
                    continue
                _, image_base64 = screen_result
                
                prompt = """请读取屏幕上背包法杖的详情面板，提取完整信息并返回JSON：
{
  "name": "法杖名称",
  "type": "法杖类型",
  "description": "详细描述",
  "attributes": "属性（如伤害、射速、散射等）",
  "effects": ["效果1","效果2"],
  "all_text": "面板上所有可见的文字（完整提取）"
}
只返回JSON，不要其他文字。"""
                print(f"  [AI分析背包法杖 {idx}...]")
                result = self.game_analyzer.analyze_image(image_base64, prompt)
                item_info = self.parse_json(result) if result else None
                if not item_info or not isinstance(item_info, dict):
                    print(f"  ✗ 解析失败（法杖 {idx}）")
                    continue
                
                wand_name = item_info.get("name", f"未知法杖{idx}") or f"未知法杖{idx}"
                # 保存到以名称为键的法杖缓存
                entry = {
                    'item_id': None,
                    'item_name': wand_name,
                    'item_info': item_info,
                    'cached_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'source': 'backpack'
                }
                self.wand_cache[f"name_{wand_name}"] = entry
                print(f"  ✓ 已识别并缓存法杖: {wand_name}")
                owned_wands.append(wand_name)
            except Exception as e:
                print(f"  ✗ 识别法杖 {idx} 失败: {e}")
        
        if owned_wands:
            # 立即保存法杖缓存
            try:
                with open(self.wand_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.wand_cache, f, indent=2, ensure_ascii=False)
                print(f"  ✓ 已保存背包法杖缓存（{len(owned_wands)}）")
            except Exception as e:
                print(f"  ⚠ 保存背包法杖缓存失败: {e}")
        # 更新已拥有摘要
        self.owned_items_summary['wands'] = owned_wands
    
    def analyze_item_info(self, item_id: int, item_name: str, item_type: int, store_items: List[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """分析商品信息（法杖或法术）"""
        # 先检查缓存（传入类型以便从对应缓存查找）
        cached_info = self.get_cached_item_info(item_id, item_type)
        if cached_info:
            print(f"  ✓ 从缓存获取商品信息: {item_name}")
            return cached_info.get('item_info')
        
        # 找到对应的货架位置
        if store_items is None:
            store_data = self.fetch_store_data()
            if not store_data:
                return None
            store_items = store_data.get('store', [])
        
        shelf_index = None
        for idx, item in enumerate(store_items):
            if item.get('id') == item_id:
                shelf_index = idx
                break
        
        if shelf_index is None:
            print(f"  ✗ 未找到商品 {item_name} 的货架位置")
            return None
        
        # 移动到货架位置
        if not self.move_to_shelf(shelf_index):
            return None
        
        # 截取全屏
        screen_result = self.capture_full_screen()
        if screen_result is None:
            return None
        
        _, image_base64 = screen_result
        
        # 根据类型构建不同的提示词
        if item_type == 0:  # 法杖
            prompt = """请仔细分析这个游戏画面中出现的商品详情窗口内的所有可见文字，包括商品名称、类型、属性、效果、数值等所有信息。
特别要把所有文字都提取出来，包括描述性文字、数值、符号等。
只返回JSON格式，不要其他文字。JSON格式如下：
{
  "name": "商品名称",
  "type": "商品类型",
  "description": "详细描述文字",
  "attributes": "所有属性（如伤害、射速、散射等）",
  "effects": ["效果1", "效果2"],
  "all_text": "窗口内所有可见的文字（完整提取）"
}
如果没有某项信息，使用空字符串""或空数组[]。注意all_text字段要提取窗口内所有文字内容。"""
        else:  # 法术 (type 1)
            prompt = """请仔细分析这个游戏画面中出现的商品详情窗口内的所有可见文字，包括法术名称、类型、效果、属性、数值等所有信息。
特别要把所有文字都提取出来，包括描述性文字、数值、符号等。
只返回JSON格式，不要其他文字。JSON格式如下：
{
  "name": "法术名称",
  "type": "法术类型",
  "spell_category": "主动" 或 "被动",
  "cooling_time": "冷却时间",
  "description": "详细描述文字",
  "damage": "伤害值",
  "cost": "消耗",
  "attributes": "所有属性（如散射角度、范围等）",
  "effects": ["效果1", "效果2"],
  "all_text": "窗口内所有可见的文字（完整提取）"
}
注意：spell_category字段非常重要，必须判断是"主动"还是"被动"法术。
如果没有某项信息，使用空字符串""或空数组[]。注意all_text字段要提取窗口内所有文字内容。"""
        
        print("  [AI分析中...]")
        result = self.game_analyzer.analyze_image(image_base64, prompt)
        
        if result:
            # 解析JSON
            item_info = self.parse_json(result)
            if item_info:
                # 保存到缓存（传入类型以便保存到对应缓存）
                self.cache_item_info(item_id, item_name, item_info, item_type)
                self.save_item_cache(item_type)
                return item_info
        
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
    
    def analyze_all_items(self, store_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分析所有需要分析的商品（法杖和法术）"""
        store_items = store_data.get('store', [])
        analyzed_items = []
        
        for item in store_items:
            item_id = item.get('id')
            item_name = item.get('name', '')
            item_type = item.get('type', -1)
            item_price = item.get('price', 0)
            
            # 只分析法杖(type 0)和法术(type 1)
            if item_type in [0, 1]:
                print(f"\n[分析商品] {item_name} (ID: {item_id}, Type: {item_type})")
                item_info = self.analyze_item_info(item_id, item_name, item_type, store_items)
                
                analyzed_item = {
                    'id': item_id,
                    'name': item_name,
                    'type': item_type,
                    'price': item_price,
                    'info': item_info
                }
                analyzed_items.append(analyzed_item)
            else:
                # 消耗品直接使用基本信息
                analyzed_item = {
                    'id': item_id,
                    'name': item_name,
                    'type': item_type,
                    'price': item_price,
                    'info': None
                }
                analyzed_items.append(analyzed_item)
        
        return analyzed_items
    
    def build_decision_prompt(self, coin: int, items: List[Dict[str, Any]], owned: Dict[str, Any] = None) -> str:
        """构建购买决策的提示词"""
        items_text = []
        for item in items:
            item_id = item.get('id')
            item_name = item.get('name', '')
            item_type = item.get('type', -1)
            item_price = item.get('price', 0)
            item_info = item.get('info')
            
            item_desc = f"- {item_name} (ID: {item_id}, 价格: {item_price}金币"
            
            if item_type == 0:
                item_desc += ", 类型: 法杖"
            elif item_type == 1:
                item_desc += ", 类型: 法术"
            elif item_type == 4:
                item_desc += ", 类型: 消耗品"
            
            if item_info:
                if item_type == 0:  # 法杖
                    item_desc += f", 属性: {item_info.get('attributes', '')}, 效果: {item_info.get('effects', [])}"
                elif item_type == 1:  # 法术
                    item_desc += f", 类别: {item_info.get('spell_category', '')}, 伤害: {item_info.get('damage', '')}, 效果: {item_info.get('effects', [])}"
            
            item_desc += ")"
            items_text.append(item_desc)
        
        items_summary = "\n".join(items_text)
        
        # 构建商品索引和价格的映射，方便计算总价
        items_with_index = []
        for idx, item in enumerate(items):
            items_with_index.append(f"索引{idx}: {item.get('name', '')} - {item.get('price', 0)}金币")
        
        items_index_summary = "\n".join(items_with_index)
        
        owned_text = ""
        if owned:
            owned_wands = owned.get('wands') or []
            owned_spells = owned.get('spells') or []
            owned_lines = []
            if owned_wands:
                owned_lines.append(f"- 已有法杖: {', '.join(owned_wands)}")
            if owned_spells:
                owned_lines.append(f"- 已有法术: {', '.join(owned_spells)}")
            if owned_lines:
                owned_text = "\n当前已拥有：\n" + "\n".join(owned_lines) + "\n"
        
        prompt = f"""以下是游戏商店中的商品信息，请你帮助判断应该购买哪些商品，并给出简短理由。

**重要约束：购买商品的总价绝对不能超过当前金币 {coin}！**

当前金币: {coin}

{owned_text if owned_text else ""}
商品列表（带索引和价格）：
{items_index_summary}

商品详细信息：
{items_summary}

商品类型说明：
- type 0 = 法杖：主要武器，影响攻击方式和伤害
- type 1 = 法术：主动或被动技能，提供各种效果
- type 4 = 消耗品：一次性使用的道具（如钥匙、护盾等）

购买策略建议：
1. **必须遵守：所选商品的总价不能超过当前金币 {coin}**
2. 优先考虑能提升战斗力的核心装备（法杖、核心法术）
3. 根据当前金币合理分配，可以保留部分金币以备后用
4. 消耗品（如钥匙、护盾）在需要时购买
5. 法杖和法术要考虑与当前构筑的配合度

**计算示例**：如果选择索引[0, 1, 3]，需要计算这三个商品的价格总和，确保不超过 {coin} 金币。

只允许输出以下JSON格式，不得添加额外内容：
{{"purchases": [0, 2, 5], "reason": "购买理由"}}
purchases是一个数组，包含要购买的商品索引（从0开始，对应商品列表的顺序），reason用一句话说明购买理由。
**请确保purchases中所有商品的价格总和不超过 {coin} 金币！**
如果不需要购买任何商品，返回空数组：{{"purchases": [], "reason": "不购买的理由"}}"""
        
        return prompt
    
    def decide_purchases(self, coin: int, items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """基于商品信息调用语言模型决策购买哪些商品"""
        decision_prompt = self.build_decision_prompt(coin, items, owned=self.owned_items_summary or {})
        # 尝试当前模型，失败后回退到通用可用模型（如 gpt-4o-mini）
        reply = None
        tried_models = []
        fallback_models = []
        if self.text_model:
            fallback_models.append(self.text_model)
        fallback_models.append("gpt-4o-mini")
        fallback_models.append("gpt-3.5-turbo")
        for mdl in fallback_models:
            if mdl in tried_models:
                continue
            tried_models.append(mdl)
            try:
                completion = self.text_client.chat.completions.create(
                    model=mdl,
                    messages=[
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": decision_prompt}],
                        }
                    ],
                )
                reply = completion.choices[0].message.content
                break
            except Exception as exc:
                print(f"✗ 调用文本模型失败(model={mdl}): {exc}")
                continue
        if reply is None:
            print("✗ 无法生成购买决策")
            return None
        
        decision = self.parse_json(reply)
        if decision is None:
            print("✗ 无法解析决策模型的JSON回复")
            print(f"原始回复: {reply}")
            return None
        
        # 验证总价是否超过当前金币
        purchases = decision.get('purchases', [])
        total_price = 0
        for idx in purchases:
            if 0 <= idx < len(items):
                total_price += items[idx].get('price', 0)
        
        if total_price > coin:
            print(f"⚠ 警告：AI建议购买的商品总价 {total_price} 超过了当前金币 {coin}，正在修正...")
            # 修正：只保留能买得起的商品
            affordable_purchases = []
            remaining_coin = coin
            for idx in purchases:
                if 0 <= idx < len(items):
                    item_price = items[idx].get('price', 0)
                    if item_price <= remaining_coin:
                        affordable_purchases.append(idx)
                        remaining_coin -= item_price
                    else:
                        break
            
            decision['purchases'] = affordable_purchases
            if affordable_purchases:
                decision['reason'] = f"（已修正）原建议总价超限，调整为购买以下商品：{', '.join([items[i].get('name', '') for i in affordable_purchases])}"
            else:
                decision['reason'] = "（已修正）原建议总价超限，建议不购买任何商品"
        
        return decision
    
    def run(self) -> None:
        """执行完整流程"""
        print("=" * 60)
        print("商店购买决策工具")
        print("=" * 60)
        
        # 0. 识别当前已拥有的法杖（用于影响购买策略）
        try:
            self.analyze_owned_wands()
        except Exception as e:
            print(f"⚠ 已有物品识别步骤失败: {e}")
        
        # 1. 获取商店数据
        print("\n[步骤1] 获取商店数据...")
        store_data = self.fetch_store_data()
        if not store_data:
            print("✗ 无法获取商店数据")
            return
        
        # 2. 获取当前金币
        print("\n[步骤2] 获取当前金币...")
        coin = self.get_current_coin()
        
        # 3. 分析所有商品（法杖和法术需要分析详情）
        print("\n[步骤3] 分析商品信息...")
        analyzed_items = self.analyze_all_items(store_data)
        
        # 4. 决策购买
        print("\n[步骤4] AI决策购买...")
        decision = self.decide_purchases(coin, analyzed_items)
        
        if not decision:
            print("✗ 无法生成购买决策")
            return
        
        print("\n决策结果:")
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        
        # 输出购买列表
        purchases = decision.get('purchases', [])
        if purchases:
            print(f"\n建议购买的商品索引: {purchases}")
            total_price = 0
            for idx in purchases:
                if 0 <= idx < len(analyzed_items):
                    item = analyzed_items[idx]
                    item_price = item.get('price', 0)
                    total_price += item_price
                    print(f"  - {item.get('name')} (价格: {item_price}金币)")
            print(f"\n总价: {total_price}金币")
            print(f"剩余金币: {coin - total_price}金币")
        else:
            print("\n建议不购买任何商品")
        
        print(f"\n理由: {decision.get('reason', '无')}")
        
        # 5. 执行购买
        if purchases:
            print("\n[步骤5] 执行购买...")
            self.execute_purchases(purchases, analyzed_items, coin)
            # 购买结束后执行法术构筑流程
            try:
                print("\n[步骤6] 执行法术构筑（spell_construct_flow.py）...")
                self.run_spell_construct_flow()
            except Exception as e:
                print(f"⚠ 执行法术构筑失败: {e}")
        
        # 保存结果
        output = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "coin": coin,
            "items": analyzed_items,
            "decision": decision,
        }
        
        with open("store_purchase_result.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print("\n✓ 结果已保存到: store_purchase_result.json")
    
    def execute_purchases(self, purchase_indices: List[int], items: List[Dict[str, Any]], current_coin: int):
        """执行购买操作：移动到对应货架位置并按E键购买"""
        print(f"\n开始执行购买 {len(purchase_indices)} 个商品...")
        
        # 获取商店数据以确定货架位置
        store_data = self.fetch_store_data()
        if not store_data:
            print("✗ 无法获取商店数据，无法执行购买")
            return
        
        store_items = store_data.get('store', [])
        
        purchased_count = 0
        total_spent = 0
        
        for idx in purchase_indices:
            if idx < 0 or idx >= len(items):
                print(f"  ⚠ 商品索引 {idx} 超出范围，跳过")
                continue
            
            item = items[idx]
            item_id = item.get('id')
            item_name = item.get('name', '')
            item_price = item.get('price', 0)
            
            # 检查金币是否足够
            if total_spent + item_price > current_coin:
                print(f"  ⚠ 金币不足，无法购买 {item_name} (需要 {item_price}，已花费 {total_spent}，剩余 {current_coin - total_spent})")
                continue
            
            print(f"\n[购买商品 {purchased_count + 1}/{len(purchase_indices)}] {item_name} (价格: {item_price}金币)")
            
            # 找到对应的货架位置
            shelf_index = None
            for i, store_item in enumerate(store_items):
                if store_item.get('id') == item_id:
                    shelf_index = i
                    break
            
            if shelf_index is None:
                print(f"  ✗ 未找到商品 {item_name} 的货架位置")
                continue
            
            # 移动到货架位置
            print(f"  [步骤1] 移动到货架 {shelf_index + 1}...")
            if not self.move_to_shelf(shelf_index):
                print(f"  ✗ 移动到货架失败，跳过购买 {item_name}")
                continue
            
            # 按E键购买
            print(f"  [步骤2] 按E键购买...")
            try:
                pydirectinput.press('e')
                time.sleep(0.3)  # 等待购买完成
                print(f"  ✓ 已按E键购买 {item_name}")
                
                purchased_count += 1
                total_spent += item_price
                
                # 更新当前金币（购买后金币会减少）
                current_coin -= item_price
                
            except Exception as e:
                print(f"  ✗ 按E键失败: {e}")
                continue
        
        print(f"\n✓ 购买完成: 成功购买 {purchased_count} 个商品，总花费 {total_spent} 金币")

    def run_spell_construct_flow(self) -> None:
        """直接调用 spell_construct_flow.py 完成交互式构筑流程"""
        cmd = [sys.executable, "spell_construct_flow.py"]
        print(f"  [调用] {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, check=True)
            if proc.returncode == 0:
                print("  ✓ spell_construct_flow 执行完成")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ spell_construct_flow 执行失败，返回码 {e.returncode}")
        except Exception as e:
            print(f"  ✗ 调用 spell_construct_flow 失败: {e}")


def main():
    decision = StorePurchaseDecision()
    decision.run()


if __name__ == "__main__":
    main()

