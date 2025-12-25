"""
实时检测脚本 - 使用YOLOv5模型实时检测游戏窗口画面
支持动态调整每个分类的阈值
"""
import os
import sys
import cv2
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import threading
import time
import queue

# 尝试导入 keyboard 库用于全局按键检测
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("警告: keyboard 库未安装，按键检测可能需要在 OpenCV 窗口有焦点时才能工作")

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置环境变量
BASE_DIR = os.path.dirname(__file__)

# 导入YOLOv5相关模块
# 注意：需要安装ultralytics或yolov5库
try:
    from ultralytics.utils.plotting import Annotator, colors
except ImportError:
    try:
        from utils.plots import Annotator, colors
    except ImportError:
        print("警告: 未找到绘图工具，可视化功能可能受限")
        Annotator = None
        colors = None

# 尝试导入YOLOv5工具函数（如果用户有yolov5目录或已安装）
try:
    # 尝试从yolov5目录导入
    yolov5_path = os.path.join(BASE_DIR, 'yolov5')
    if os.path.isdir(yolov5_path) and yolov5_path not in sys.path:
        sys.path.insert(0, yolov5_path)
    from models.common import DetectMultiBackend
    from utils.general import (
        non_max_suppression,
        scale_boxes,
        check_img_size,
    )
    from utils.augmentations import letterbox
    from utils.torch_utils import select_device
    YOLOV5_AVAILABLE = True
except ImportError:
    YOLOV5_AVAILABLE = False
    print("警告: 未找到YOLOv5库，请确保已安装或yolov5目录存在")
    print("提示: 可以使用 pip install ultralytics 或下载yolov5到项目目录")
from get_game_window import GameWindowCapture

# 自动控制相关模块
try:
    from control_thread import ControlThread
    from auto_attack import AutoAttack
    from auto_dodge import AutoDodge
except ImportError:
    ControlThread = None
    AutoAttack = None
    AutoDodge = None

try:
    # 门处理与后续流程
    from door_handler import DoorHandler
except ImportError:
    DoorHandler = None

try:
    # 商店购买决策
    from store_purchase_decision import StorePurchaseDecision
except ImportError:
    StorePurchaseDecision = None

try:
    # 法术构筑流程
    from spell_construct_flow import run as spell_construct_flow_run
except ImportError:
    spell_construct_flow_run = None


class SimpleLogger:
    """提供给 AutoDodge / DoorHandler 使用的简易 logger"""

    def __init__(self):
        import logging

        self._logger = logging.getLogger("realtime_agent")

    @property
    def logger(self):
        return self._logger

    def log(self, msg: str, level: str = "info"):
        text = f"[{level}] {msg}"
        print(text)
        if hasattr(self._logger, level):
            getattr(self._logger, level)(msg)
        else:
            self._logger.info(msg)


class RealtimeDetector:
    """实时检测器类"""
    
    def __init__(
        self,
        weights_path: str,
        data_yaml_path: str,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        device: str = '',
        imgsz: int = 640
    ):
        """
        初始化检测器
        
        Args:
            weights_path: 模型权重文件路径
            data_yaml_path: 数据配置文件路径
            conf_thres: 默认置信度阈值
            iou_thres: IOU阈值
            device: 设备 ('cuda' 或 'cpu')
            imgsz: 输入图像尺寸
        """
        self.weights_path = weights_path
        self.data_yaml_path = data_yaml_path
        self.iou_thres = iou_thres
        self.imgsz = imgsz

        # 读取类别信息
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        self.class_names = data_config.get('names', {})
        self.num_classes = len(self.class_names)

        # 为每个类别初始化阈值（默认使用统一阈值）
        self.class_thresholds = {i: conf_thres for i in range(self.num_classes)}
        
        # 加载模型
        self.device = select_device(device)
        self.model = DetectMultiBackend(
            weights=weights_path,
            device=self.device,
            dnn=False,
            data=data_yaml_path,
            fp16=False
        )
        self.stride = self.model.stride
        self.names = self.model.names

        # 检查输入尺寸
        self.imgsz = check_img_size(imgsz, s=self.stride)

        # 初始化窗口捕获器
        self.capturer = GameWindowCapture()
        self.is_running = False
        self.detection_thread = None
        self.input_thread = None
        self.input_queue = queue.Queue()
        self.frame_count = 0

        # 控制与自动行为
        self.control_thread: Optional["ControlThread"] = None
        self.auto_attack: Optional["AutoAttack"] = None
        self.auto_dodge: Optional["AutoDodge"] = None
        self.logger = SimpleLogger()

        # RGB颜色检测开关（默认关闭以提升性能）
        self.enable_color_detection = False

        # 窗口偏移（用于把检测坐标转换为屏幕坐标）
        self.window_offset_x = 0
        self.window_offset_y = 0

        # 门与房间状态
        self.current_room_type: Optional[str] = None  # 例如: "store" / "unknown"
        self.last_door_enter_time: float = 0.0

        # 图片保存相关
        self.save_dir = os.path.join(BASE_DIR, "saved_images")
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_counter = 0
        self.current_result_frame = None  # 保存当前结果帧，用于按键保存

        print(f"模型加载完成: {weights_path}")
        print(f"设备: {self.device}")
        print(f"类别数量: {self.num_classes}")
        print(f"图片保存目录: {self.save_dir}")
    
    def set_class_threshold(self, class_id: int, threshold: float):
        """设置单个类别的阈值"""
        if 0 <= class_id < self.num_classes:
            self.class_thresholds[class_id] = max(0.0, min(1.0, threshold))
            print(f"类别 {class_id} ({self.class_names.get(class_id, f'class{class_id}')}) 阈值设置为: {self.class_thresholds[class_id]:.3f}")
        else:
            print(f"无效的类别ID: {class_id}")
    
    def set_all_thresholds(self, threshold: float):
        """设置所有类别的阈值"""
        for class_id in range(self.num_classes):
            self.class_thresholds[class_id] = max(0.0, min(1.0, threshold))
        print(f"所有类别阈值设置为: {threshold:.3f}")
    
    def get_class_threshold(self, class_id: int) -> float:
        """获取单个类别的阈值"""
        return self.class_thresholds.get(class_id, 0.25)
    
    def detect_with_class_thresholds(
        self,
        image: np.ndarray,
        class_thresholds: Dict[int, float]
    ) -> Tuple[torch.Tensor, Tuple, Tuple]:
        """
        使用每个类别不同的阈值进行检测
        
        Args:
            image: 输入图像 (BGR格式)
            class_thresholds: 每个类别的阈值字典
            
        Returns:
            (检测结果张量, ratio, pad) - ratio和pad用于坐标缩放
        """
        # 使用letterbox预处理图像（保持宽高比）
        img, ratio, pad = letterbox(image, self.imgsz, stride=self.stride, auto=True)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        
        # 推理
        pred = self.model(img, augment=False, visualize=False)
        
        # 使用最小的阈值进行初步过滤（保留更多候选框）
        min_threshold = min(class_thresholds.values())
        pred = non_max_suppression(
            pred,
            conf_thres=min_threshold,
            iou_thres=self.iou_thres,
            classes=None,
            agnostic=False,
            max_det=1000
        )[0]
        
        # 根据每个类别的阈值进行二次过滤
        if len(pred) > 0:
            filtered_detections = []
            for det in pred:
                cls = int(det[5])
                conf = float(det[4])
                cls_thresh = class_thresholds.get(cls, min_threshold)
                if conf >= cls_thresh:
                    filtered_detections.append(det)
            
            if len(filtered_detections) > 0:
                pred = torch.stack(filtered_detections)
            else:
                pred = torch.empty((0, 6), device=pred.device)
        
        # 将坐标缩放回原图尺寸
        if len(pred) > 0:
            h, w = image.shape[:2]
            # scale_boxes 需要 img.shape[2:] 作为模型输入尺寸，ratio_pad 为 (ratio, pad)
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], (h, w), ratio_pad=(ratio, pad)).round()
        
        return pred, ratio, pad

    def _detect_bullets_by_color(self, frame: np.ndarray, color_tolerance: int = 25) -> List[Dict[str, Any]]:
        """
        使用RGB颜色范围检测子弹（优化版本，使用降采样和合并掩码）
        
        Args:
            frame: 输入图像 (BGR格式)
            color_tolerance: 颜色容差值，允许RGB值上下浮动的范围
            
        Returns:
            检测到的子弹列表
        """
        # 定义两个目标RGB颜色（BGR格式，因为OpenCV使用BGR）
        target_colors = [
            (42, 40, 55),   # RGB(55,40,42) -> BGR(42,40,55)
            (13, 17, 185),  # RGB(185,17,13) -> BGR(13,17,185)
        ]
        
        # 降采样以提高性能（缩放到原来的一半）
        scale_factor = 0.5
        h, w = frame.shape[:2]
        small_h, small_w = int(h * scale_factor), int(w * scale_factor)
        small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
        
        bullets = []
        
        # 合并所有颜色范围的掩码，一次性处理（更高效）
        combined_mask = np.zeros((small_h, small_w), dtype=np.uint8)
        
        for target_bgr in target_colors:
            # 计算颜色范围
            lower = np.array([
                max(0, target_bgr[0] - color_tolerance),
                max(0, target_bgr[1] - color_tolerance),
                max(0, target_bgr[2] - color_tolerance)
            ], dtype=np.uint8)
            upper = np.array([
                min(255, target_bgr[0] + color_tolerance),
                min(255, target_bgr[1] + color_tolerance),
                min(255, target_bgr[2] + color_tolerance)
            ], dtype=np.uint8)
            
            # 创建颜色掩码并合并
            mask = cv2.inRange(small_frame, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # 查找轮廓（只处理一次）
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 对每个轮廓，如果大小合理则认为是子弹
        for contour in contours:
            area = cv2.contourArea(contour)
            # 过滤太小的区域（可能是噪声），注意这是在降采样图像上的面积
            # 需要按scale_factor^2转换为原图面积来判断
            original_area = area / (scale_factor ** 2)
            if original_area < 10 or original_area > 5000:
                continue
            
            # 获取边界框（需要缩放回原图尺寸）
            x, y, bw, bh = cv2.boundingRect(contour)
            # 缩放回原图坐标
            x1, y1 = float(x / scale_factor), float(y / scale_factor)
            x2, y2 = float((x + bw) / scale_factor), float((y + bh) / scale_factor)
            
            # 计算中心点
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            
            bullet_item = {
                "class_id": 2,  # bullet
                "class_name": "bullet",
                "confidence": 0.8,  # RGB检测的置信度
                "bbox": (x1, y1, x2, y2),
                "center": center,
                "detection_method": "color"  # 标记为颜色检测
            }
            bullets.append(bullet_item)
        
        return bullets
    
    def _build_objects(self, detections: torch.Tensor, frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        将 YOLO 检测结果转换为便于决策使用的对象集合
        可选地使用RGB颜色检测来补充子弹检测
        
        Args:
            detections: YOLO检测结果
            frame: 原始图像帧（可选，用于RGB颜色检测）
        """
        objs: Dict[str, Any] = {
            "mc": None,
            "monsters": [],
            "bullets": [],
            "traps": [],
            "spells": [],
            "rewards": [],
            "boxes": [],
            "coins": [],
            "doors": [],
        }

        # 处理YOLO检测结果
        if detections is not None and len(detections) > 0:
            for det in detections:
                x1, y1, x2, y2, conf, cls = det.tolist()
                cls = int(cls)
                bbox = (x1, y1, x2, y2)
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                item = {
                    "class_id": cls,
                    "class_name": self.class_names.get(cls, str(cls)),
                    "confidence": float(conf),
                    "bbox": bbox,
                    "center": center,
                    "detection_method": "yolo"  # 标记为YOLO检测
                }

                # 按当前 data.yaml 中的类别约定进行分类
                if cls == 11:  # mc
                    objs["mc"] = item
                elif cls == 12:  # monster
                    objs["monsters"].append(item)
                elif cls == 2:  # bullet
                    objs["bullets"].append(item)
                elif cls == 20:  # trap
                    objs["traps"].append(item)
                elif cls == 18:  # spell
                    objs["spells"].append(item)
                elif cls in (0, 5, 6, 16, 19):  # boss_reward, gold_reward, health_reward, relic_reward, spellbook_reward
                    objs["rewards"].append(item)
                elif cls in (8, 9, 10):  # lockedbox*, hurt/cursed
                    objs["boxes"].append(item)
                elif cls == 3:  # coin
                    objs["coins"].append(item)
                elif cls == 4:  # door
                    objs["doors"].append(item)

        # 使用RGB颜色检测补充子弹检测（仅在启用时）
        if frame is not None and self.enable_color_detection:
            color_bullets = self._detect_bullets_by_color(frame)
            
            # 将颜色检测到的子弹与YOLO检测到的子弹合并
            # 如果颜色检测到的子弹与YOLO检测到的距离很近（30像素内），认为是同一个，只保留YOLO的结果
            for color_bullet in color_bullets:
                color_center = color_bullet["center"]
                # 检查是否与已有子弹太近
                is_duplicate = False
                for existing_bullet in objs["bullets"]:
                    existing_center = existing_bullet["center"]
                    dist = np.sqrt((color_center[0] - existing_center[0])**2 + 
                                 (color_center[1] - existing_center[1])**2)
                    if dist < 30:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    objs["bullets"].append(color_bullet)

        return objs

    def draw_detections(self, image: np.ndarray, detections: torch.Tensor) -> np.ndarray:
        """在图像上绘制检测结果"""
        annotator = Annotator(image, line_width=2, example=str(self.names))
        
        if len(detections) > 0:
            for det in detections:
                *xyxy, conf, cls = det
                c = int(cls)  # integer class
                label = f"{self.names[c]} {conf:.2f}"
                annotator.box_label(xyxy, label, color=colors(c, True))
        
        return annotator.result()

    def _move_towards(self, mc: Dict[str, Any], target: Dict[str, Any], step_key_duration: float = 0.12):
        """用 WASD 简单朝向目标移动一点（每次调用只发送一键）"""
        if not self.control_thread:
            return
        mc_x, mc_y = mc["center"]
        tx, ty = target["center"]
        dx = tx - mc_x
        dy = ty - mc_y

        key: Optional[str] = None
        # 优先水平靠近，其次垂直
        if abs(dx) > abs(dy):
            key = "d" if dx > 0 else "a"
        else:
            key = "s" if dy > 0 else "w"
        #
        self.control_thread.press_key(key, duration=step_key_duration)

    @staticmethod
    def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)

    def _handle_auto_actions(self, frame: np.ndarray, detections: torch.Tensor):
        """根据当前检测结果执行自动行为（躲避、攻击、拾取、门逻辑）"""
        if self.control_thread is None or AutoAttack is None or AutoDodge is None:
            return

        objs = self._build_objects(detections, frame)
        mc = objs["mc"]
        if not mc:
            return

        self.frame_count += 1
        frame_shape = frame.shape

        # 1 & 6: 自动远离 monster / bullet / trap
        if self.auto_dodge:
            threats_for_dodge = list(objs["monsters"]) + list(objs["traps"])
            self.auto_dodge.set_frame_count(self.frame_count)
            self.auto_dodge.execute(mc, threats_for_dodge, objs["bullets"], frame_shape)

        # 2: 自动锁定最近 monster，移动鼠标并按 K 攻击
        if self.auto_attack and objs["monsters"]:
            self.auto_attack.set_frame_count(self.frame_count)
            self.auto_attack.execute(objs["monsters"], mc=mc)

        # 3~7: 没有紧急威胁时，优先去捡东西 / 开箱 / 拾金币
        has_threat = bool(objs["monsters"] or objs["bullets"] or objs["traps"])
        if not has_threat:
            pickups: List[Dict[str, Any]] = []
            # 按优先级：spell > reward > box > coin
            pickups.extend(objs["spells"])
            pickups.extend(objs["rewards"])
            pickups.extend(objs["boxes"])
            pickups.extend(objs["coins"])

            if pickups:
                mc_center = mc["center"]
                # 最近的目标
                target = min(pickups, key=lambda it: self._distance(mc_center, it["center"]))
                dist = self._distance(mc_center, target["center"])

                # 若距离较远则先走过去
                if dist > 40:
                    self._move_towards(mc, target)
                else:
                    # 到达附近，根据类型按 E 或让游戏自动捡起
                    cls_id = target["class_id"]
                    if cls_id in (18, 0, 5, 6, 16, 19, 8, 9, 10):  # spell / 奖励 / 宝箱
                        self.control_thread.press_key("e", duration=0.08)
                    # coin 只需要走过去即可自动拾取

        # 8~10: 简化版门逻辑：清空 3/4/5/7 里的目标（spell/reward/box/coin）后再考虑进门
        if objs["doors"]:
            # 是否还有需要处理的 3~7 目标
            has_pending_pickups = bool(
                objs["spells"] or objs["rewards"] or objs["boxes"] or objs["coins"]
            )
            if not has_pending_pickups:
                # 选择最近的门
                door = min(objs["doors"], key=lambda it: self._distance(mc["center"], it["center"]))
                dist_door = self._distance(mc["center"], door["center"])

                # 先走近门（阈值稍微放宽一点，避免永远达不到）
                if dist_door > 80:
                    self._move_towards(mc, door)
                else:
                    print(f"[流程] 门附近，开始执行进门前流程，距离约 {dist_door:.1f}px")
                    # 进入门前，执行一次法术构筑流程
                    if spell_construct_flow_run is not None:
                        print("[流程] 进入门前执行 spell_construct_flow.run()")
                        try:
                            spell_construct_flow_run()
                        except Exception as e:
                            print(f"[流程] 执行 spell_construct_flow.run() 失败: {e}")

                    # 使用 E 进门
                    print("[流程] 靠近门，按 E 进入")
                    self.control_thread.press_key("e", duration=0.1)
                    self.last_door_enter_time = time.time()
                    self.current_room_type = None

                    # 简单判定是否为商店：调用商店接口看是否有数据
                    if StorePurchaseDecision is not None:
                        try:
                            print("[流程] 进门后尝试调用 StorePurchaseDecision 判断是否为商店...")
                            decider = StorePurchaseDecision()
                            store_data = decider.fetch_store_data()
                            if store_data and store_data.get("store"):
                                self.current_room_type = "store"
                                print("[流程] 判定为商店房间，执行商店购买与相关流程")
                                decider.run()
                            else:
                                self.current_room_type = "unknown"
                                print("[流程] 当前房间非商店或接口无数据，跳过商店流程")
                        except Exception as e:
                            self.current_room_type = "unknown"
                            print(f"[流程] StorePurchaseDecision 流程失败: {e}")

                    # 如有 DoorHandler，可在此记录和恢复自动控制（这里仅简单调用）
                    if DoorHandler is not None:
                        try:
                            dummy_controller = type(
                                "DummyController",
                                (),
                                {
                                    "auto_aim_enabled": True,
                                    "auto_dodge_enabled": True,
                                    "auto_attack_enabled": True,
                                },
                            )()
                            door_handler = DoorHandler(self.logger)
                            door_handler.near_door = True
                            door_handler.handle_door_sequence(mc, door, dummy_controller)
                        except Exception as e:
                            print(f"[流程] DoorHandler.handle_door_sequence 执行失败: {e}")

    def _detection_loop(self):
        """检测循环"""
        print("开始检测循环...")
        fps_counter = 0
        fps_start_time = time.time()
        
        while self.is_running:
            # 获取最新帧
            frame = self.capturer.get_latest_frame()
            
            if frame is not None:
                start_time = time.time()
                
                # 进行检测
                detections, ratio, pad = self.detect_with_class_thresholds(frame, self.class_thresholds)

                # 基于检测结果执行自动行为
                self._handle_auto_actions(frame, detections)

                # 绘制检测结果
                result_frame = self.draw_detections(frame.copy(), detections)
                
                # 计算FPS
                inference_time = time.time() - start_time
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                else:
                    fps = 1.0 / inference_time if inference_time > 0 else 0
                
                # 显示FPS和检测数量
                cv2.putText(
                    result_frame,
                    f"FPS: {fps:.1f} | Detections: {len(detections)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # 显示结果
                cv2.imshow("Realtime Detection", result_frame)
                
                # 保存当前结果帧，供按键保存使用
                self.current_result_frame = result_frame.copy()
                
                # 打印检测信息（减少输出频率）
                # if len(detections) > 0:
                #     for det in detections:
                #         cls = int(det[5])
                #         conf = float(det[4])
                #         print(f"检测到: {self.names[cls]} (置信度: {conf:.3f}, 阈值: {self.class_thresholds[cls]:.3f})")
            
            # 处理输入队列中的命令
            try:
                while True:
                    command = self.input_queue.get_nowait()
                    self._process_command(command)
            except queue.Empty:
                pass
            
            # 处理按键
            # 使用 keyboard 库进行全局按键检测（不需要窗口焦点）
            if KEYBOARD_AVAILABLE:
                if keyboard.is_pressed("q"):
                    # 按 Q 键保存图片
                    if self.current_result_frame is not None:
                        self._save_image(self.current_result_frame)
                        # 等待按键释放，避免重复触发
                        while keyboard.is_pressed("q"):
                            time.sleep(0.01)
                    else:
                        print("无法保存：当前没有可用的图片帧")
                # elif keyboard.is_pressed("escape"):
                #     # ESC 键退出
                #     self.is_running = False
                #     break
                elif keyboard.is_pressed("r"):
                    # 重置所有阈值为0.25
                    self.set_all_thresholds(0.25)
                    while keyboard.is_pressed("r"):
                        time.sleep(0.01)
                elif keyboard.is_pressed("h"):
                    # 显示帮助信息
                    self._print_help()
                    while keyboard.is_pressed("h"):
                        time.sleep(0.01)
            else:
                # 回退到 cv2.waitKey（需要窗口有焦点）
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # 255 表示没有按键按下
                    if key == 27:  # ESC 键退出
                        self.is_running = False
                        break
                    elif key == ord('q') or key == ord('Q'):
                        # 按 Q 键保存图片
                        print("检测到 Q 键按下...")
                        if self.current_result_frame is not None:
                            self._save_image(self.current_result_frame)
                        else:
                            print("无法保存：当前没有可用的图片帧")
                    elif key == ord('r') or key == ord('R'):
                        # 重置所有阈值为0.25
                        self.set_all_thresholds(0.25)
                    elif key == ord('h') or key == ord('H'):
                        # 显示帮助信息
                        self._print_help()
            # 让 OpenCV 处理窗口消息
            cv2.waitKey(1)
        
        cv2.destroyAllWindows()
        print("检测循环结束")
    
    def _save_image(self, frame: np.ndarray):
        """保存当前帧图片"""
        if frame is None:
            print("无法保存：当前帧为空")
            return
        
        self.save_counter += 1
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}_{self.save_counter:04d}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        
        cv2.imwrite(filepath, frame)
        print(f"图片已保存: {filepath}")

    def _print_help(self):
        """打印帮助信息"""
        print("\n" + "="*50)
        print("=== 阈值调整方法 ===")
        print("="*50)
        print("\n【方法1】在代码中设置（启动前）:")
        print("  在 main() 函数中，创建检测器后添加：")
        print("  detector.set_class_threshold(类别ID, 阈值)")
        print("  例如: detector.set_class_threshold(0, 0.5)")
        print("\n【方法2】运行时通过控制台命令调整（推荐）:")
        print("  在检测运行时，直接在控制台输入以下命令：")
        print("\n  命令列表:")
        print("    set <类别ID> <阈值>  - 设置指定类别的阈值")
        print("                         例如: set 0 0.5  (将类别0设为0.5)")
        print("                         例如: set 5 0.7  (将类别5设为0.7)")
        print("    setall <阈值>        - 设置所有类别的阈值")
        print("                         例如: setall 0.3  (所有类别都设为0.3)")
        print("    show                 - 显示所有类别的当前阈值")
        print("    help                 - 显示此帮助信息")
        print("\n【快捷键】:")
        print("    ESC - 退出检测")
        print("    q - 保存当前图片")
        print("    r - 重置所有阈值到0.25")
        print("    h - 显示此帮助信息")
        print("\n当前类别阈值:")
        for class_id in range(self.num_classes):
            class_name = self.class_names.get(class_id, f"class{class_id}")
            threshold = self.class_thresholds[class_id]
            print(f"  {class_id:2d}. {class_name:15s}: {threshold:.3f}")
        print("=" * 20 + "\n")
    
    def _process_command(self, command: str):
        """处理控制台命令"""
        parts = command.strip().split()
        if not parts:
            return
        
        cmd = parts[0].lower()
        
        if cmd == "set" and len(parts) == 3:
            try:
                class_id = int(parts[1])
                threshold = float(parts[2])
                self.set_class_threshold(class_id, threshold)
            except ValueError:
                print(f"无效的命令格式: {command}")
                print("正确格式: set <类别ID> <阈值>")
        elif cmd == "setall" and len(parts) == 2:
            try:
                threshold = float(parts[1])
                self.set_all_thresholds(threshold)
            except ValueError:
                print(f"无效的命令格式: {command}")
                print("正确格式: setall <阈值>")
        elif cmd == "show":
            print("\n当前类别阈值:")
            for class_id in range(self.num_classes):
                class_name = self.class_names.get(class_id, f"class{class_id}")
                threshold = self.class_thresholds[class_id]
                print(f"  {class_id:2d}. {class_name:15s}: {threshold:.3f}")
            print()
        elif cmd == "help":
            self._print_help()
        else:
            print(f"未知命令: {command}")
            print("输入 'help' 查看帮助信息")
    
    def _input_loop(self):
        """控制台输入循环"""
        print("\n控制台输入线程已启动，可以在控制台输入命令调整阈值")
        print("输入 'help' 查看帮助信息\n")
        
        while self.is_running:
            try:
                command = input().strip()
                if command:
                    self.input_queue.put(command)
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                print(f"输入处理错误: {e}")
    
    def start(self, window_title: Optional[str] = None, fps: int = 30):
        """开始实时检测"""
        if self.is_running:
            print("检测已在运行中")
            return
        
        # 选择窗口
        if window_title:
            if not self.capturer.select_window(window_title=window_title):
                print(f"无法找到窗口: {window_title}")
                return
        else:
            if not self.capturer.select_magicraft_window():
                print("无法找到Magicraft游戏窗口")
                return
        
        window_info = self.capturer.get_window_info()
        print(f"成功选择窗口: {window_info.get('title', 'Unknown')}")
        print(f"窗口尺寸: {window_info.get('width', 0)}x{window_info.get('height', 0)}")

        # 计算窗口左上角偏移，用于把检测坐标转换为屏幕坐标
        self.window_offset_x = int(window_info.get("left", 0))
        self.window_offset_y = int(window_info.get("top", 0))

        # 初始化控制线程和自动行为模块
        if ControlThread is not None:
            self.control_thread = ControlThread()
            self.control_thread.start()
        else:
            self.control_thread = None
            print("警告: 未找到 ControlThread，自动控制功能不可用")

        if self.control_thread and AutoAttack is not None:
            self.auto_attack = AutoAttack(
                self.control_thread, self.window_offset_x, self.window_offset_y
            )
        else:
            self.auto_attack = None

        if self.control_thread and AutoDodge is not None:
            self.auto_dodge = AutoDodge(self.control_thread, self.logger)
        else:
            self.auto_dodge = None

        # 开始捕获
        self.capturer.start_capture(fps=fps)
        time.sleep(1)  # 等待捕获开始
        
        # 开始检测
        self.is_running = True
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        # 启动输入线程（用于接收控制台命令）
        self.input_thread = threading.Thread(target=self._input_loop)
        self.input_thread.daemon = True
        self.input_thread.start()
        
        print("\n实时检测已启动")
        self._print_help()
        
        # 等待检测线程结束
        try:
            self.detection_thread.join()
        except KeyboardInterrupt:
            print("\n接收到中断信号，正在停止...")
            self.stop()
    
    def stop(self):
        """停止实时检测"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.capturer.stop_capture()

        if self.control_thread:
            try:
                self.control_thread.stop()
            except Exception as e:
                print(f"停止控制线程失败: {e}")

        if self.detection_thread:
            self.detection_thread.join(timeout=2)
        
        print("实时检测已停止")


# select_device 已从 utils.torch_utils 导入，无需重复定义


def main():
    """主函数"""
    # 配置路径 - 从环境变量或默认路径读取
    base_dir = Path(__file__).parent
    weights_path = Path(os.getenv('MODEL_WEIGHTS_PATH', 
        str(base_dir / "weights" / "best.pt")))
    data_yaml_path = Path(os.getenv('MODEL_DATA_YAML', 
        str(base_dir / "data.yaml")))
    
    # 检查文件是否存在
    if not weights_path.exists():
        print(f"错误: 模型文件不存在: {weights_path}")
        print("提示: 请设置环境变量 MODEL_WEIGHTS_PATH 指定模型路径，或将模型文件放在 weights/best.pt")
        return
    
    if not data_yaml_path.exists():
        print(f"错误: 数据配置文件不存在: {data_yaml_path}")
        print("提示: 请设置环境变量 MODEL_DATA_YAML 指定配置文件路径，或将配置文件放在 data.yaml")
        return
    
    # 创建检测器
    detector = RealtimeDetector(
        weights_path=str(weights_path),
        data_yaml_path=str(data_yaml_path),
        conf_thres=0.25,
        iou_thres=0.45,
        device='',
        imgsz=640
    )
    
    # ===== 在这里调整每个分类的阈值 =====
    # 阈值范围：0.0 - 1.0，数值越大要求置信度越高（检测越严格）
    # 取消下面对应类别的注释并修改阈值来调整检测灵敏度
    
    # 打印所有类别信息，方便用户查看
    print("\n" + "="*60)
    print("所有类别列表（可在下方代码中调整阈值）：")
    print("="*60)
    for class_id in range(len(detector.class_names)):
        class_name = detector.class_names.get(class_id, f"class{class_id}")
        print(f"  类别ID {class_id:2d}: {class_name}")
    print("="*60)
    print("\n提示：取消下方代码中对应类别的注释并修改阈值即可调整")
    print("例如：detector.set_class_threshold(0, 0.5)  # 将boss_reawrd的阈值设为0.5\n")
    
    # 所有类别列表（共{}个类别）：""".format(len(detector.class_names))
    detector.set_class_threshold(0, 0.1)   # boss_reawrd - 默认阈值0.25
    detector.set_class_threshold(1, 0.25)   # breakable - 默认阈值0.25
    detector.set_class_threshold(2, 0.3)   # bullet - 默认阈值0.25
    detector.set_class_threshold(3, 0.1)   # coin - 默认阈值0.25
    detector.set_class_threshold(4, 0.25)   # door - 默认阈值0.25
    detector.set_class_threshold(5, 0.1)   # gold_reward - 默认阈值0.25
    detector.set_class_threshold(6, 0.1)   # health_reward - 默认阈值0.25
    detector.set_class_threshold(7, 0.25)   # jitan - 默认阈值0.25
    detector.set_class_threshold(8, 0.25)   # lockedbox - 默认阈值0.25
    detector.set_class_threshold(9, 0.05)   # lockedbox_cursed - 默认阈值0.25
    detector.set_class_threshold(10, 0.25)  # lockedbox_hurt - 默认阈值0.25
    detector.set_class_threshold(11, 0.5)  # mc - 默认阈值0.25
    detector.set_class_threshold(12, 0.8)  # monster - 默认阈值0.25
    detector.set_class_threshold(13, 0.25)  # objects - 默认阈值0.25
    detector.set_class_threshold(14, 0.1)  # opened_box - 默认阈值0.25
    detector.set_class_threshold(15, 0.1)  # potion - 默认阈值0.25
    detector.set_class_threshold(16, 0.1)  # relic_reward - 默认阈值0.25
    detector.set_class_threshold(17, 0.25)  # remove_curse - 默认阈值0.25
    detector.set_class_threshold(18, 0.1)  # spell - 默认阈值0.25
    detector.set_class_threshold(19, 0.1)  # spellbook_reward - 默认阈值0.25
    detector.set_class_threshold(20, 0.7)  # trap - 默认阈值0.25
    
    # 或者设置所有类别为同一个阈值：
    # detector.set_all_thresholds(0.25)  # 所有类别都设置为0.25
    
    # 开始实时检测
    try:
        detector.start(window_title=None, fps=30)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        detector.stop()


if __name__ == "__main__":
    main()

