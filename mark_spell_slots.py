from typing import List, Dict, Tuple
import json
import os
from pathlib import Path

import cv2
import numpy as np
import urllib.request
import urllib.error
from dotenv import load_dotenv

# 加载.env文件
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

from get_game_window import GameWindowCapture


def load_rows_from_json(json_path: str) -> List[List[Dict[str, int]]]:
    """
    加载 slots_all.json 中的所有行，按照键名顺序（row1, row2, row3...）加载。
    支持两种结构：
      1) {"row1": {"normal_slots": [...], "post_slots": [...]}, ...}
      2) {"row1": [...], "row2": [...]}（旧格式）
    返回的 rows[0] 对应 row1（背包），rows[1] 对应 row2（法杖1），以此类推。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows: List[List[Dict[str, int]]] = []
    # 按照键名顺序（row1, row2, row3...）加载
    sorted_keys = sorted([k for k in data.keys() if k.startswith("row")], key=lambda k: int(k[3:]) if k[3:].isdigit() else 999)
    for key in sorted_keys:
        val = data[key]
        row_list: List[Dict[str, int]] = []
        if isinstance(val, dict) and isinstance(val.get("normal_slots"), list):
            row_list = val.get("normal_slots") or []
        elif isinstance(val, list):
            row_list = val
        if row_list:
            row_sorted = sorted(row_list, key=lambda d: d.get("x", 0))
            rows.append(row_sorted)
    return rows


def split_normal_post(row: List[Dict[str, int]], gap_threshold: int = 120) -> Tuple[List[Dict[str, int]], List[Dict[str, int]]]:
    if not row:
        return [], []
    # Find first large gap between consecutive x to separate groups (only for counting/split)
    xs = [p["x"] for p in row]
    split_idx = None
    for i in range(1, len(xs)):
        if xs[i] - xs[i - 1] >= gap_threshold:
            split_idx = i
            break
    if split_idx is None:
        return row, []
    return row[:split_idx], row[split_idx:]


def generate_positions_from_counts(
    row: List[Dict[str, int]],
    normal_count: int,
    post_count: int,
    dx_normal: int = 72,
    gap_normal_to_post: int = 147,
) -> Tuple[List[Dict[str, int]], List[Dict[str, int]]]:
    """
    Use ONLY the first item's position as anchor and generate evenly spaced positions.
    Do not reuse individual detected circle x positions; just the counts.
    """
    if not row:
        return [], []
    # anchor from the first item
    anchor = row[0]
    ax, ay, ar = int(anchor["x"]), int(anchor["y"]), int(anchor.get("r", 28))

    normal = []
    for i in range(normal_count):
        normal.append({"x": ax + i * dx_normal, "y": ay, "r": ar})

    post = []
    if post_count > 0:
        start_post_x = ax + (normal_count - 1) * dx_normal + gap_normal_to_post
        for j in range(post_count):
            post.append({"x": start_post_x + j * dx_normal, "y": ay, "r": ar})

    return normal, post


def draw_slots(image: np.ndarray, points: List[Dict[str, int]], color: Tuple[int, int, int], label: str):
    for i, p in enumerate(points):
        x, y, r = int(p["x"]), int(p["y"]), int(p.get("r", 28))
        cv2.circle(image, (x, y), max(10, r), color, 2)
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        cv2.putText(image, f"{label}{i+1}", (x - 14, y - r - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def fetch_spell_counts(endpoint: str = None) -> Tuple[List[Tuple[int, int]], int]:
    """获取法术数量，从环境变量读取默认端点"""
    if endpoint is None:
        endpoint = os.getenv('SPELLS_ENDPOINT', 'http://localhost:1234/spells')
    """
    返回：
      - wands_counts: 列表[(normal_count, post_count), ...]，长度等于实际检测到的法杖数量
      - bag_count: 背包槽位数量
    失败时返回 ([], 0)
    """
    try:
        with urllib.request.urlopen(endpoint, timeout=1.5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        bag = payload.get("Bag", [])
        wands = payload.get("Wands", [])
        bag_count = len(bag) if isinstance(bag, list) else 0

        wands_counts: List[Tuple[int, int]] = []
        if isinstance(wands, list):
            for w in wands:
                if not isinstance(w, dict):
                    wands_counts.append((0, 0))
                    continue
                normal_count = len(w.get("normal_slots", [])) if isinstance(w.get("normal_slots", []), list) else 0
                post_count = len(w.get("post_slots", [])) if isinstance(w.get("post_slots", []), list) else 0
                wands_counts.append((normal_count, post_count))

        return wands_counts, bag_count
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError):
        return [], 0


def main():
    json_path = "slots_all.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON not found: {json_path}. Run detect_circle.py first.")

    rows = load_rows_from_json(json_path)

    # Build summary as required and for drawing
    # Try to get authoritative counts from the running service
    wands_counts, bag_cnt = fetch_spell_counts()
    bag_cnt = bag_cnt or 0
    wand_total = len(wands_counts)

    # 期望的行数 = 1行背包 + N行法杖（N 为实际检测到的法杖数量）
    expected_rows = 1 + wand_total

    # 固定 Y 坐标规则：row1=63（背包），row2=163，row3=263，...，row8=763（每行间距100像素）
    # 找到最后一个非空行作为模板（用于 X 坐标）
    template_row = None
    for i in range(len(rows) - 1, -1, -1):
        if rows[i] and len(rows[i]) > 0:
            template_row = rows[i]
            break
    
    if template_row is None:
        # 如果所有行都是空的，使用默认模板
        template_row = [{"x": 180 + i * 72, "y": 163, "r": 28} for i in range(4)]
    
    # 确保所有行都有正确的 Y 坐标，并补充缺失的行
    for idx in range(expected_rows):
        target_y = 63 + idx * 100  # row1=63, row2=163, row3=263, ...
        
        if idx >= len(rows):
            # 需要生成新行
            new_row = []
            for p in template_row:
                new_row.append({"x": p["x"], "y": target_y, "r": p.get("r", 28)})
            rows.append(new_row)
        elif rows[idx] and len(rows[idx]) > 0:
            # 修正现有行的 Y 坐标
            for p in rows[idx]:
                p["y"] = target_y
        elif not rows[idx] or len(rows[idx]) == 0:
            # 空行，生成坐标
            new_row = []
            for p in template_row:
                new_row.append({"x": p["x"], "y": target_y, "r": p.get("r", 28)})
            rows[idx] = new_row

    summary = {}
    split_rows = []

    # 只处理实际需要的行数（1行背包 + 实际法杖数量）
    for idx in range(expected_rows):
        if idx >= len(rows):
            # 如果行数不足，跳过（应该不会到这里，因为前面已经补充了）
            normal, post = [], []
        else:
            row = rows[idx]
            # idx==0 背包，其余对应法杖 idx-1
            if idx == 0:
                normal_cnt, post_cnt = bag_cnt, 0
                # 背包如果没有权威数量，回退到检测拆分
                if normal_cnt == 0:
                    normal_detected, post_detected = split_normal_post(row, gap_threshold=120)
                    normal_cnt = len(normal_detected)
                    post_cnt = len(post_detected)
            else:
                wand_idx = idx - 1
                if 0 <= wand_idx < len(wands_counts):
                    normal_cnt, post_cnt = wands_counts[wand_idx]
                else:
                    normal_cnt, post_cnt = 0, 0
                # 法杖行：如果权威数量为0，直接返回空，不进行回退检测
                # 这样可以避免没有法杖时还生成数据

            normal, post = generate_positions_from_counts(
                row=row,
                normal_count=normal_cnt,
                post_count=post_cnt,
                dx_normal=72,
                gap_normal_to_post=147,
            )
        split_rows.append((normal, post))
        key = f"row{idx+1}"
        summary[key] = {"normal_slots": normal, "post_slots": post}

    # Capture a frame from the game window
    capturer = GameWindowCapture()
    # Prefer selecting by title; fallback to default selection
    if not capturer.select_window(window_title="Magicraft"):
        if not capturer.select_window():
            raise RuntimeError("未找到可用窗口，请确保游戏正在运行。")
    frame = capturer.capture_frame()
    if frame is None:
        raise RuntimeError("无法捕获窗口画面。")

    # Draw markers
    out = frame.copy()
    colors = {
        "normal": (0, 255, 0),   # green
        "post": (0, 165, 255),   # orange
    }

    for idx, (normal, post) in enumerate(split_rows):
        row_label_n = f"R{idx+1}-N"
        row_label_p = f"R{idx+1}-P"
        draw_slots(out, normal, colors["normal"], row_label_n)
        draw_slots(out, post, colors["post"], row_label_p)

    cv2.imwrite("spell_slots_marked.png", out)
    print("Saved: spell_slots_marked.png")

    # 将结果写入 slots_all.json 覆盖
    output_json_path = "slots_all.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output_json_path}")

    # Print concise summary
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

