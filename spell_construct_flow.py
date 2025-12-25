import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import urllib.request
import urllib.error
import pyautogui
import pydirectinput
import cv2
from openai import OpenAI
from dotenv import load_dotenv

# 加载.env文件
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

from get_game_window import GameWindowCapture
from capture_and_analyze import GameAnalyzer
from omni_models.omni import get_text_client, get_text_model
from mark_spell_slots import load_rows_from_json, generate_positions_from_counts, fetch_spell_counts
from utils.paths import DATA_DIR


# 从环境变量读取配置
SPELLS_ENDPOINT = os.getenv('SPELLS_ENDPOINT', 'http://localhost:1234/spells')
WAND_CACHE_FILE = "store_wand_cache.json"
SLOTS_JSON = "slots_all.json"
DANGEROUS_SPELL_NAMES = {"诡雷"}
DANGEROUS_KEYWORDS = ("无差别伤害", "自伤", "反弹", "爆炸")
PROJECTILE_KEYWORDS = ["法术飞弹", "魔法弹", "Magic Missile", "蝴蝶", "彩虹", "激光", "落雷", "黑洞", "冥蛇", "滚石", "诡雷", "瓦解射线", "注魔硬币", "审判之剑", "次元行者"]

THINK_MODEL_NAME = os.getenv('THINK_MODEL', 'deepseek/deepseek-v3.2-think')
THINK_CLIENT = OpenAI(
    base_url=os.getenv('BASE_URL', 'https://router.shengsuanyun.com/api/v1'),
    api_key=os.getenv('API_KEY', 'hApEHVmHSZm1-4NHWfPWfUy9boJ4ZngnqUlJJ3_AYErSyAMNKDAg6TSeLx7-_9NXiFzAT2LArA'),
)


def fetch_spells_payload(endpoint: str = SPELLS_ENDPOINT) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(endpoint, timeout=1.5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError):
        return None


def is_dangerous_spell_info(info: Optional[Dict[str, Any]], fallback_name: str = "") -> bool:
    """
    检测法术是否存在自伤风险：
    - 名称命中危险名单（如诡雷）
    - 描述/效果包含“无差别伤害”“反弹”等关键词
    """
    name = ""
    text_parts: List[str] = []
    if isinstance(info, dict):
        name = str(info.get("name", "") or fallback_name or "")
        effects = info.get("effects", [])
        if isinstance(effects, list):
            text_parts.extend([str(e) for e in effects])
        text_parts.append(str(info.get("description", "") or ""))
        text_parts.append(str(info.get("all_text", "") or ""))
        text_parts.append(str(info.get("attributes", "") or ""))
    else:
        name = str(fallback_name or "")
    if name in DANGEROUS_SPELL_NAMES:
        return True
    check_text = " ".join(text_parts + [name])
    for kw in DANGEROUS_KEYWORDS:
        if kw and kw in check_text:
            return True
    return False


def is_projectile_spell(name: str) -> bool:
    if not name:
        return False
    return any(kw in name for kw in PROJECTILE_KEYWORDS)


def is_guardian_wand_spirit(spell_name: str) -> bool:
    """检查是否是守护杖灵系列法术（固定法术，不应移动）"""
    if not spell_name:
        return False
    name = str(spell_name).strip()
    return name in ("守护杖灵", "守护杖灵+", "守护杖灵++")


def load_wand_cache(path: Path = Path(WAND_CACHE_FILE)) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def summarize_wand_for_prompt(wand_cache: Dict[str, Any], service_wands: List[Dict[str, Any]], wand_idx: int) -> str:
    lines: List[str] = []
    if 0 <= wand_idx - 1 < len(service_wands):
        w = service_wands[wand_idx - 1]
        if isinstance(w, dict):
            lines.append(f"- {w.get('name','未知')}: max_mp={w.get('max_mp')}, mp_recover={w.get('mp_recover')}, shoot_interval={w.get('shoot_interval')}, cooldown={w.get('cooldown')}")
    detected = wand_cache.get("detected_wands") or []
    for entry in detected:
        if not isinstance(entry, dict):
            continue
        if entry.get("wand_index") == wand_idx:
            panel = entry.get("panel_info") or {}
            special = (panel.get("description") or panel.get("attributes") or "").strip()
            name = entry.get("item_name") or (entry.get("basic", {}) or {}).get("name") or "未知法杖"
            if special:
                lines.append(f"- {name} 额外描述: {special}")
    return "\n".join(lines)


def get_wand_slot_counts(wand_cache: Dict[str, Any], wand_idx: int) -> Tuple[Optional[int], Optional[int]]:
    normal_c = None
    post_c = None
    detected = (wand_cache.get("detected_wands") or [])
    for entry in detected:
        if not isinstance(entry, dict):
            continue
        if entry.get("wand_index") != wand_idx:
            continue
        slots = entry.get("slots") or {}
        normal = (slots.get("normal") or {}).get("positions") or []
        post = (slots.get("post") or {}).get("positions") or []
        try:
            normal_c = len(normal)
        except Exception:
            pass
        try:
            post_c = len(post)
        except Exception:
            pass
        break
    return normal_c, post_c


def summarize_equipped_spells(service_wands: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    if not isinstance(service_wands, list):
        return ""
    for i, w in enumerate(service_wands):
        if not isinstance(w, dict):
            continue
        wand_name = w.get("name", f"法杖{i+1}")
        normal_slots = w.get("normal_slots", [])
        if not isinstance(normal_slots, list):
            continue
        for j, slot in enumerate(normal_slots):
            if isinstance(slot, dict):
                lines.append(f"- {wand_name} 槽位{j+1}: {slot.get('name', '')} (ID: {slot.get('id', '')})")
    return "\n".join(lines)


def parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    import re
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m2 = re.search(r"(\{.*\})", text, re.DOTALL)
    if m2:
        try:
            return json.loads(m2.group(1))
        except Exception:
            pass
    return None


def run_think_completion(prompt: str) -> Tuple[Optional[str], str]:
    """
    调用思考模型并流式输出 reasoning_content，返回(最终回答, 思考文本)。
    """
    try:
        completion = THINK_CLIENT.chat.completions.create(
            model=THINK_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            top_p=0.7,
            stream=True,
        )
        reasoning_text = ""
        answer_text = ""
        for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "reasoning_content", None):
                reasoning_piece = delta.reasoning_content
                reasoning_text += reasoning_piece
                print(f"[thought] {reasoning_piece}", end="", flush=True)
            if delta.content is not None:
                answer_piece = delta.content
                answer_text += answer_piece
                print(answer_piece, end="", flush=True)
        print()  # ensure newline after streaming
        return answer_text.strip(), reasoning_text.strip()
    except Exception as exc:
        print(f"✗ 调用思考模型失败: {exc}")
        return None, ""


def run_text_completion_streamed(prompt: str) -> Optional[str]:
    """
    调用普通文本模型的流式接口，边收边打印最终回答。
    """
    client = get_text_client()
    model = get_text_model()
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            stream=True,
        )
        answer_text = ""
        for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content_piece = ""
            if isinstance(delta.content, str):
                content_piece = delta.content
            elif isinstance(delta.content, list):
                # 新接口 content 可能是 list[ContentPart]
                for part in delta.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        content_piece += str(part.get("text", ""))
                    elif hasattr(part, "text"):
                        content_piece += str(getattr(part, "text", ""))
            if content_piece:
                answer_text += content_piece
                print(content_piece, end="", flush=True)
        print()  # newline after stream
        return answer_text.strip()
    except Exception as exc:
        print(f"✗ 调用文本模型失败: {exc}")
        return None


def decide_with_text_model(prompt: str, use_think_model: bool = False) -> Optional[Dict[str, Any]]:
    reply: Optional[str] = None
    reasoning_text: str = ""
    use_think_model = False
    try:
        if use_think_model:
            reply, reasoning_text = run_think_completion(prompt)
        else:
            reply = run_text_completion_streamed(prompt)
    except Exception as exc:
        print(f"✗ 调用文本模型失败: {exc}")
        return None
    if reply is None:
        print("✗ 模型未返回内容")
        return None
    if use_think_model and reasoning_text:
        print("\n--- 收到的思考过程 ---")
        print(reasoning_text.strip())
        print("\n--- 最终回答 ---")
        print(reply.strip())
    decision = parse_json(reply)
    if decision is None:
        print("✗ 无法解析文本模型的JSON回复")
        print(f"原始回复: {reply}")
        return None
    return decision


def build_construct_prompt(
    wand_summary: str,
    equipped_summary: str,
    candidate_spells_summary: List[Dict[str, Any]],
    target_count: int = 4,
    normal_count: Optional[int] = None,
    post_count: Optional[int] = None,
) -> str:
    summary_text = json.dumps(candidate_spells_summary, ensure_ascii=False, indent=2)
    constraints = []
    if normal_count is not None:
        constraints.append(f"- 可装备的 normal 槽位数量: {normal_count}")
    if post_count is not None:
        constraints.append(f"- 法杖自带的 post 槽位数量: {post_count}（自动触发，不与 normal 槽互动）")
    slots_constraints = "\n".join(constraints) if constraints else "（槽位数量未知）"
    print(slots_constraints)
    prompt = f"""请基于当前法杖与已装备法术的描述，从背包法术中选择最优的{target_count}个法术用于构筑，并说明理由。

当前法杖（含基础与特殊描述）：
{wand_summary if wand_summary else "无"}

当前已装备法术：
{equipped_summary if equipped_summary else "无"}

槽位约束：
{slots_constraints}

可选法术池（背包 + 当前法杖已有，index从1开始）：
{summary_text}
注意：每个法术条目中的 "available_count" 字段表示该法术名的总可用数量（背包+两杖已装备）。布局中同名法术的出现次数不能超过其 available_count。

规则与建议：
1. 总目标：在保证基本生存的前提下，优先最大化持续输出（秒伤）。防御型法术优先级较低（目前可靠的纯防御法术很少）。
2. normal 槽与 post 槽互不作用；post 槽为法杖自触发效果（如移动/站立充能等），可以自动释放，最好放入法术飞弹或法术召唤等主动法术。
3. **重要位置规则（只针对 normal 槽）**：
   - 被动/增益在左，主动输出在右
   - 法术增强等增益必须位于目标主动法术左侧才生效
   - **特别重要：法术增强（如伤害强化、分裂、多重施法等）必须放在法术飞弹（如蝴蝶、魔法弹、法术飞弹等）的左侧，否则增益无效**
4. 选择核心输出：在可用 normal 槽内，必须包含核心输出（如黑洞、落雷阵、冥蛇、诡雷、滚石、激光、蝴蝶、注魔硬币等）。
5. 魔法弹（法术飞弹/Magic Missile）为"纯白版"时仅是过渡输出。只有在存在明确增益/命中提升（如寻踪、自动导航、分裂、多重射击、伤害强化、法术增强等）时才可作为核心或进入布局；否则降低其优先级，优先考虑更强主动输出（如蝴蝶/落雷阵/黑洞/滚石/诡雷等）。
6. 构筑思路：
   - 单法杖体系：适合槽位/法术较少时，集中堆叠增益在一个核心法术上，优先确保稳定命中与持续输出能力。
   - 多法杖体系：仅当存在共鸣/杖灵/魔导书等"多法杖启动"条件时再考虑，以同时启动多根法杖法术为目标。
7. 控制类法术（如冰冻）可少量纳入以提升安全性，但不得挤占核心输出与关键增益的位置。
8. 允许同名法术在不同槽位重复出现，但必须遵守 available_count 限制：布局中每个法术名的出现次数不能超过其 available_count。
9. 最终请给出当前法杖 normal 槽从左到右的法术名布局，严格遵守上面的生效位置规则与优先级。布局长度必须等于可用 normal 槽位数量（即 {normal_count if normal_count is not None else 'N'} 个），用尽所有可用槽；若候选不足，才可重复同名（但不超过 available_count）。

重要规则：
1. 尽量避免使用诡雷，那个东西会对自身造成伤害
2. 分裂最好搭配闪电链组合，然后主动的法术飞弹最好选择蝴蝶或彩虹这类散射的法术

只返回JSON：{{"layout": ["法术名1","法术名2","法术名3","法术名4"], "reason": "一句话理由"}}"""
    print(prompt)
    return prompt


def build_global_construct_prompt(
    wand_cache: Dict[str, Any],
    wands_from_service: List[Dict[str, Any]],
    bag_spells: List[Dict[str, Any]],
    equipped_summary: str,
) -> Dict[str, Any]:
    """
    为两根法杖一起构筑的提示词与辅助信息。
    返回:
      {
        "prompt": str,
        "wand_normals": {1: normal_cnt1, 2: normal_cnt2},
      }
    """
    wand_infos: List[Dict[str, Any]] = []
    wand_normals: Dict[int, int] = {}

    # 聚合全局可用数量（背包 + 两根法杖）
    global_name_to_count: Dict[str, int] = {}
    for s in bag_spells:
        info = s.get("spell_info", {}) or {}
        nm = info.get("name", "未知")
        if is_dangerous_spell_info(info, nm):
            continue
        if is_guardian_wand_spirit(nm):
            continue  # 跳过守护杖灵系列法术
        if nm:
            global_name_to_count[nm] = global_name_to_count.get(nm, 0) + 1
    for w in wands_from_service:
        if isinstance(w, dict):
            eq_slots = w.get("normal_slots", []) or []
            for slot_obj in eq_slots:
                if isinstance(slot_obj, dict):
                    nm = slot_obj.get("name", "")
                    if is_dangerous_spell_info(None, nm):
                        continue
                    if is_guardian_wand_spirit(nm):
                        continue  # 跳过守护杖灵系列法术
                    if nm:
                        global_name_to_count[nm] = global_name_to_count.get(nm, 0) + 1

    # 每根法杖的信息（包含已占用槽位信息）
    for wand_idx, w in enumerate(wands_from_service, start=1):
        if not isinstance(w, dict):
            continue
        ws = summarize_wand_for_prompt(wand_cache, wands_from_service, wand_idx)
        n_cnt, p_cnt = get_wand_slot_counts(wand_cache, wand_idx)
        service_n_cnt = None
        ns = w.get("normal_slots", [])
        if isinstance(ns, list):
            service_n_cnt = len(ns)
        effective_n_cnt = service_n_cnt if (isinstance(service_n_cnt, int) and service_n_cnt > 0) else (n_cnt if (isinstance(n_cnt, int) and n_cnt > 0) else 0)
        wand_normals[wand_idx] = effective_n_cnt
        
        # 获取已占用槽位信息（特别是守护杖灵的位置）
        occupied_slots: List[Dict[str, Any]] = []
        for slot_idx, slot_obj in enumerate(ns, start=1):
            if isinstance(slot_obj, dict):
                slot_name = slot_obj.get("name", "")
                if slot_name and is_guardian_wand_spirit(slot_name):
                    occupied_slots.append({
                        "slot_index": slot_idx,
                        "spell_name": slot_name,
                        "is_fixed": True  # 固定法术，不可移动
                    })
        
        wand_infos.append(
            {
                "wand_index": wand_idx,
                "wand_name": w.get("name", f"法杖{wand_idx}"),
                "normal_slots": effective_n_cnt,
                "post_slots": p_cnt or 0,
                "summary": ws or "",
                "occupied_slots": occupied_slots,  # 已占用的槽位（如守护杖灵）
            }
        )

    # 候选法术池：背包 + 两根法杖当前已装备
    candidates: List[Dict[str, Any]] = []
    # bag
    for i, s in enumerate(bag_spells, start=1):
        info = s.get("spell_info", {}) or {}
        nm = info.get("name", "未知")
        if is_dangerous_spell_info(info, nm):
            continue
        if is_guardian_wand_spirit(nm):
            continue  # 跳过守护杖灵系列法术（固定法术，不应参与构筑）
        candidates.append(
            {
                "index": i,
                "source": "bag",
                "name": nm,
                "type": info.get("type", ""),
                "category": info.get("spell_category", ""),
                "attributes": info.get("attributes", ""),
                "effects": info.get("effects", []),
                "description": (info.get("all_text", "") or "")[:160],
                "available_count": global_name_to_count.get(nm, 0),
            }
        )
    # equipped on each wand
    for wand_idx, w in enumerate(wands_from_service, start=1):
        if not isinstance(w, dict):
            continue
        eq_slots = w.get("normal_slots", []) or []
        for slot_obj in eq_slots:
            if isinstance(slot_obj, dict):
                nm = slot_obj.get("name", "未知")
                if is_dangerous_spell_info(None, nm):
                    continue
                if is_guardian_wand_spirit(nm):
                    continue  # 跳过守护杖灵系列法术（固定法术，不应参与构筑）
                candidates.append(
                    {
                        "index": len(candidates) + 1,
                        "source": f"wand{wand_idx}",
                        "name": nm,
                        "type": "",
                        "category": "已装备",
                        "attributes": "",
                        "effects": [],
                        "description": f"ID: {slot_obj.get('id','')}（当前法杖{wand_idx}已装备）",
                        "available_count": global_name_to_count.get(nm, 0),
                    }
                )

    summary_text = json.dumps(candidates, ensure_ascii=False, indent=2)
    wand_info_text = json.dumps(wand_infos, ensure_ascii=False, indent=2)

    # 生成所有法杖的键名
    wand_keys = [f"wand{i}" for i in range(1, len(wand_infos) + 1)]
    wand_keys_json = ", ".join([f'"{k}"' for k in wand_keys])
    
    prompt = f"""你是一个构筑助手，需要在**全局视角**下一次性为所有法杖规划法术布局。

当前所有法杖的信息（normal 槽数量、post 槽数量、面板总结）如下：
{wand_info_text}

当前已装备法术汇总（所有法杖）： 
{equipped_summary if equipped_summary else "无"}

可选法术池（背包 + 所有法杖当前已装备，available_count 是全局可用总数）：
{summary_text}

全局规则与约束：
1. 你需要**同时**为所有法杖规划 normal 槽布局，输出所有法杖的布局（{wand_keys_json}）。
1.1 **每个法杖的第一个 normal 槽位固定是“守护杖灵/守护杖灵+/守护杖灵++”，绝对不要移动或覆盖，必须跳过槽位1。**
2. 对于任意法术名 X，其在所有法杖布局中的总出现次数，不能超过该法术在 candidates 中的 available_count。
3. 每根法杖的 normal 槽位数量必须与上面 wand_infos 中的 normal_slots 一致，且尽量全部用满；若槽位数为 0，则对应布局可为空数组。
4. **重要：每根法杖的 occupied_slots 字段显示了已占用的槽位（如守护杖灵），这些槽位不可使用，必须跳过。**
5. **重要位置规则（只针对 normal 槽）**：
   - 被动/增益在左，主动输出在右
   - **法术增强等增益必须位于目标主动法术左侧才生效**
   - **特别重要：法术增强（如伤害强化、分裂、多重施法等）必须放在法术飞弹（如蝴蝶、魔法弹、法术飞弹等）的左侧，否则增益无效**
    - **至少保证每根法杖有1个法术飞弹类输出，并放在最右侧的可用槽位**
6. 总目标：在保证基本生存的前提下，最大化整体持续输出（所有法杖综合秒伤），可以允许不同法杖有不同的定位（辅助/输出）。
7. 尽量避免使用"诡雷"，只有在没有其它更安全、更稳定的输出方案时才可以考虑；如果有其它合适的核心输出（如落雷阵、蝴蝶等），应优先选择其它输出法术而不是诡雷。
8. "分裂"最好搭配"闪电链"组合使用，并尽量配合蝴蝶或彩虹这类散射的主动法术，以提升多段命中与连锁收益。
9. 拟态、二重奏、回响等注意他和目标法术的位置，拟态魔方要放在法术左侧
**重要：你必须为每个法术指定具体的槽位位置（slot_index，从1开始），跳过已占用的槽位（occupied_slots）。**

只返回 JSON，格式如下（必须包含所有法杖的布局，每个法术必须指定槽位位置）：
{{
  {", ".join([f'"{k}": {{"layout": [{{"spell_name": "法术名1", "slot_index": 2}}, {{"spell_name": "法术名2", "slot_index": 3}}], "reason": "一句话说明构筑思路"}}' for k in wand_keys])}
}}

注意：layout 中的每个元素必须是对象，包含 spell_name 和 slot_index。slot_index 必须跳过 occupied_slots 中已占用的槽位。
每个法杖的layout都得有东西"""

    return {"prompt": prompt, "wand_normals": wand_normals, "global_name_to_count": global_name_to_count}


def sanitize_global_decision(
    global_decision: Dict[str, Any],
    global_name_to_count: Dict[str, int],
    wand_normals: Dict[int, int],
) -> Dict[str, Any]:
    """
    基于全局配额（global_name_to_count）对模型给出的全局布局做二次裁剪：
    保证所有法杖中每个法术名的总出现次数不超过 available_count。
    若模型超量使用，将按法杖顺序（wand1, wand2, ...）优先保留布局。
    不做自动补位，宁可少放也不超用。
    保留原始格式（对象数组或字符串数组）。
    """

    def get_layout_and_format(dec: Dict[str, Any], key: str) -> Tuple[List[str], bool, List[Any]]:
        """
        返回: (法术名列表, 是否为对象格式, 原始布局列表)
        """
        sub = dec.get(key)
        if isinstance(sub, dict):
            lay = sub.get("layout", [])
            if isinstance(lay, list):
                result_names = []
                is_object_format = False
                for item in lay:
                    if isinstance(item, dict):
                        # 新格式：对象包含 spell_name 和 slot_index
                        is_object_format = True
                        spell_name = item.get("spell_name", "")
                        if spell_name:
                            result_names.append(str(spell_name))
                    elif isinstance(item, str) and item:
                        # 旧格式：字符串数组
                        result_names.append(str(item))
                return result_names, is_object_format, lay
        return [], False, []

    # 获取所有法杖的布局
    max_wand_idx = max(wand_normals.keys()) if wand_normals else 2
    layouts: Dict[int, List[str]] = {}
    layout_formats: Dict[int, bool] = {}  # 记录是否为对象格式
    original_layouts: Dict[int, List[Any]] = {}  # 保存原始布局
    result_layouts: Dict[int, List[str]] = {}
    
    for wand_idx in range(1, max_wand_idx + 1):
        key = f"wand{wand_idx}"
        names, is_obj, orig = get_layout_and_format(global_decision, key)
        layouts[wand_idx] = names
        layout_formats[wand_idx] = is_obj
        original_layouts[wand_idx] = orig
        result_layouts[wand_idx] = []
    
    used: Dict[str, int] = {}

    # 按法杖顺序处理，优先保证前面的法杖
    for wand_idx in sorted(layouts.keys()):
        max_slots = wand_normals.get(wand_idx, len(layouts[wand_idx]) or 0)
        if max_slots <= 0:
            continue
        for nm in layouts[wand_idx]:
            total_allowed = global_name_to_count.get(nm, 0)
            if total_allowed <= 0:
                continue
            if used.get(nm, 0) >= total_allowed:
                continue
            if len(result_layouts[wand_idx]) >= max_slots:
                break
            result_layouts[wand_idx].append(nm)
            used[nm] = used.get(nm, 0) + 1

    # 回写到 global_decision 中，保留原始格式
    for wand_idx in sorted(layouts.keys()):
        key = f"wand{wand_idx}"
        sub = global_decision.get(key)
        if not isinstance(sub, dict):
            sub = {}
            global_decision[key] = sub
        
        if layout_formats[wand_idx]:
            # 对象格式：从原始布局中提取对应的对象
            result_objects = []
            orig = original_layouts[wand_idx]
            result_names_set = set(result_layouts[wand_idx])
            for item in orig:
                if isinstance(item, dict):
                    spell_name = item.get("spell_name", "")
                    if spell_name in result_names_set:
                        result_objects.append(item)
                        result_names_set.remove(spell_name)
                        if not result_names_set:
                            break
            sub["layout"] = result_objects
        else:
            # 字符串格式：直接使用结果
            sub["layout"] = result_layouts[wand_idx]

    return global_decision


# ---- Spell analysis cache (to avoid re-reading every time) ----

SPELL_CACHE_FILE = DATA_DIR / "spell_analysis_cache.json"

def load_spell_cache() -> Dict[str, Any]:
    try:
        with open(SPELL_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

def save_spell_cache(spell_cache: Dict[str, Any]) -> None:
    try:
        with open(SPELL_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(spell_cache, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def sanitize_spell_cache_categories(spell_cache: Dict[str, Any]) -> bool:
    """
    Fix misclassified enhancement spells as passive.
    Targets: 包含“法术增强”或名称为“分裂”“伤害强化”的法术。
    Returns True if any changes were made.
    """
    changed = False
    def fix_info(info: Dict[str, Any]) -> bool:
        nonlocal changed
        if not isinstance(info, dict):
            return False
        name = str(info.get("name", "") or "")
        if ("法术增强" in name) or (name in ("分裂", "伤害强化")):
            if info.get("spell_category") != "被动":
                info["spell_category"] = "被动"
                changed = True
            t = str(info.get("type", "") or "")
            if "主动" in t:
                info["type"] = t.replace("主动", "被动")
                changed = True
        return changed
    for k, v in list(spell_cache.items()):
        if isinstance(v, dict):
            # entry dict style
            if "spell_info" in v and isinstance(v["spell_info"], dict):
                fix_info(v["spell_info"])
            else:
                fix_info(v)
    return changed

def get_cached_spell_info(spell_cache: Dict[str, Any], spell_id: Optional[int] = None, spell_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    # Prefer ID
    if spell_id is not None:
        key = f"id_{spell_id}"
        cached = spell_cache.get(key)
        if isinstance(cached, dict):
            return cached.copy()
    # Then by name
    if spell_name:
        key = f"name_{spell_name}"
        cached = spell_cache.get(key)
        if isinstance(cached, str) and cached.startswith("id_"):
            cached = spell_cache.get(cached, {})
        if isinstance(cached, dict):
            return cached.copy()
    return None

def cache_spell_info(spell_cache: Dict[str, Any], spell_id: Optional[int], spell_name: Optional[str], spell_info: Dict[str, Any]) -> None:
    if spell_info is None:
        return
    # Ensure '法术增强' treated as passive
    if isinstance(spell_info, dict):
        nm = spell_info.get("name", "") or (spell_name or "")
        if "法术增强" in nm:
            spell_info["spell_category"] = "被动"
            if "type" in spell_info and "主动" in str(spell_info.get("type", "")):
                spell_info["type"] = str(spell_info["type"]).replace("主动", "被动")
    entry = {
        "spell_id": spell_id,
        "spell_name": spell_name,
        "spell_info": spell_info,
        "cached_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    if spell_id is not None:
        spell_cache[f"id_{spell_id}"] = entry.copy()
    if spell_name:
        if spell_id is not None:
            spell_cache[f"name_{spell_name}"] = f"id_{spell_id}"
        else:
            spell_cache[f"name_{spell_name}"] = entry.copy()


def client_to_screen(capturer: GameWindowCapture, x: int, y: int) -> Tuple[int, int]:
    try:
        client_rect = capturer._get_client_abs_rect()
    except Exception:
        client_rect = None
    if client_rect:
        left, top = client_rect[0], client_rect[1]
    else:
        info = capturer.get_window_info()
        left, top = info.get("left", 0), info.get("top", 0)
    return left + int(x), top + int(y)


def capture_spell_region(capturer: GameWindowCapture) -> Optional[Any]:
    # Capture full window frame then crop top-left area similar to analyze_spells
    frame = capturer.capture_frame()
    if frame is None:
        return None
    h, w = frame.shape[:2]
    crop_w = min(1000, w)
    crop_h = min(500, h)
    return frame[0:crop_h, 0:crop_w]


def analyze_spell_at(analyzer: GameAnalyzer, capturer: GameWindowCapture, sx: int, sy: int) -> Optional[Dict[str, Any]]:
    try:
        pyautogui.moveTo(sx, sy, duration=0.2)
        
        # 单击左键
        pydirectinput.click(sx, sy)
        time.sleep(1)
        cropped = capture_spell_region(capturer)
        if cropped is None:
            return None
        prompt = """请仔细分析这个法术面板中的所有可见文字，包括法术名称、类型、效果、属性、数值等所有信息。
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
        img_b64 = GameAnalyzer.image_to_base64_static(cropped) if hasattr(GameAnalyzer, "image_to_base64_static") else GameAnalyzer().image_to_base64(cropped)
        result = analyzer.analyze_image(img_b64, prompt)
        if not result:
            return None
        try:
            return json.loads(result)
        except Exception:
            parsed = parse_json(result)
            return parsed
    except Exception:
        return None


def run():
    print("=" * 60)
    print("法术构筑流程（独立测试版）")
    print("=" * 60)

    pyautogui.PAUSE = 0.1
    pyautogui.FAILSAFE = False
    pydirectinput.PAUSE = 0.01

    capturer = GameWindowCapture()
    # Prefer selecting by title; fallback to default
    if not capturer.select_window(window_title="Magicraft"):
        if not capturer.select_window():
            raise RuntimeError("未找到可用窗口，请确保游戏正在运行。")
    analyzer = GameAnalyzer()

    # Load cache
    spell_cache: Dict[str, Any] = load_spell_cache()
    if sanitize_spell_cache_categories(spell_cache):
        save_spell_cache(spell_cache)

    wand_cache = load_wand_cache()
    payload = fetch_spells_payload() or {}
    wands_from_service = payload.get("Wands", []) or []
    bag_from_service = payload.get("Bag", []) or []

    # Compute bag slot positions from slots_all.json + service counts
    rows = load_rows_from_json(SLOTS_JSON)
    counts = fetch_spell_counts()
    # 兼容新返回格式 (wands_counts, bag_cnt) 以及旧格式 ((w1_n,w1_p),(w2_n,w2_p),bag_cnt)
    wands_counts: List[Tuple[int, int]] = []
    bag_cnt = 0
    if isinstance(counts, tuple) and len(counts) == 2 and isinstance(counts[0], list):
        wands_counts, bag_cnt = counts
    elif isinstance(counts, tuple) and len(counts) == 3:
        # 旧格式回退
        (w1_n, w1_p), (w2_n, w2_p), bag_cnt = counts
        wands_counts = [(w1_n, w1_p), (w2_n, w2_p)]
    else:
        wands_counts = []
        bag_cnt = 0
    bag_count = bag_cnt or 0
    # 运行前强制刷新槽位坐标，确保行数足够
    try:
        from mark_spell_slots import main as mark_slots_main
        print("\nℹ 正在重新生成槽位坐标（mark_spell_slots）...")
        mark_slots_main()
        rows = load_rows_from_json(SLOTS_JSON)
    except Exception as e:
        print(f"⚠ 重新生成槽位坐标失败: {e}")
        rows = load_rows_from_json(SLOTS_JSON)

    if not rows or len(rows) < 1:
        raise FileNotFoundError(f"未能从 {SLOTS_JSON} 读取坐标行")
    first_row = rows[0]
    bag_normal, _ = generate_positions_from_counts(
        row=first_row,
        normal_count=bag_count,
        post_count=0,
        dx_normal=72,
        gap_normal_to_post=147,
    )
    bag_targets_screen: List[Dict[str, int]] = []
    for p in bag_normal:
        sx, sy = client_to_screen(capturer, int(p["x"]), int(p["y"]))
        bag_targets_screen.append({"x": sx, "y": sy})

    # Ensure new spells (bag and equipped) have OCR descriptions cached
    def ensure_descriptions_cached() -> None:
        # Bag
        for i, slot in enumerate(bag_from_service):
            if not isinstance(slot, dict):
                continue
            spell_id = slot.get("id")
            spell_name = slot.get("name", "")
            cached = get_cached_spell_info(spell_cache, spell_id=spell_id, spell_name=spell_name)
            if cached:
                continue
            if i < len(bag_targets_screen):
                sx, sy = bag_targets_screen[i]["x"], bag_targets_screen[i]["y"]
                info = analyze_spell_at(analyzer, capturer, sx, sy)
                if info:
                    cache_spell_info(spell_cache, spell_id, spell_name, info)
        # Equipped on each wand
        for widx, w in enumerate(wands_from_service, start=1):
            if not isinstance(w, dict):
                continue
            normal_slots_srv = w.get("normal_slots", [])
            if not isinstance(normal_slots_srv, list) or len(normal_slots_srv) == 0:
                continue
            # build targets for this wand row using service count (inline to avoid forward-ref)
            try:
                rows_local = load_rows_from_json(SLOTS_JSON)
                row_idx_local = 1 if widx == 1 else 2
                row_local = rows_local[row_idx_local]
                normal_local, _ = generate_positions_from_counts(
                    row=row_local,
                    normal_count=len(normal_slots_srv),
                    post_count=0,
                    dx_normal=72,
                    gap_normal_to_post=147,
                )
                targets_eq = [{"x": client_to_screen(capturer, int(p["x"]), int(p["y"]))[0],
                               "y": client_to_screen(capturer, int(p["x"]), int(p["y"]))[1]} for p in normal_local]
            except Exception:
                targets_eq = []
            for j, s in enumerate(normal_slots_srv):
                if not isinstance(s, dict):
                    continue
                spell_id = s.get("id")
                spell_name = s.get("name", "")
                cached = get_cached_spell_info(spell_cache, spell_id=spell_id, spell_name=spell_name)
                if cached:
                    continue
                if j < len(targets_eq):
                    coord = targets_eq[j]
                    sx, sy = coord.get("x", 0), coord.get("y", 0)
                    info = analyze_spell_at(analyzer, capturer, int(sx), int(sy))
                    if info:
                        cache_spell_info(spell_cache, spell_id, spell_name, info)

    ensure_descriptions_cached()

    # Analyze bag spells present (using service to decide which indices contain spells)
    # Also build a service-name -> index map for fallback
    bag_spells: List[Dict[str, Any]] = []
    for i, slot in enumerate(bag_from_service):
        if slot is None:
            continue
        if i >= len(bag_targets_screen):
            break
        sx, sy = bag_targets_screen[i]["x"], bag_targets_screen[i]["y"]
        spell_id = None
        spell_name = ""
        if isinstance(slot, dict):
            spell_id = slot.get("id")
            spell_name = slot.get("name", "")
        cached = get_cached_spell_info(spell_cache, spell_id=spell_id, spell_name=spell_name)
        spell_info = None
        if cached and isinstance(cached, dict):
            if "spell_info" in cached and isinstance(cached["spell_info"], dict):
                spell_info = cached["spell_info"]
            elif "name" in cached or "spell_category" in cached:
                spell_info = cached
        if spell_info is None:
            spell_info = analyze_spell_at(analyzer, capturer, sx, sy)
            if spell_info:
                cache_spell_info(spell_cache, spell_id, spell_name, spell_info)
        if spell_info:
            dangerous_flag = is_dangerous_spell_info(spell_info, spell_name)
            bag_spells.append({
                "index": len(bag_spells) + 1,
                "spell_info": spell_info,
                "coordinate": {"x": sx, "y": sy},
                "dangerous": dangerous_flag,
            })
        time.sleep(0.2)
    # Save updated cache
    save_spell_cache(spell_cache)

    equipped_summary = summarize_equipped_spells(wands_from_service)

    # Print current wand and its spells if available
    def get_current_wand_index(wand_cache_obj: Dict[str, Any], service_wands_obj: List[Dict[str, Any]]) -> Optional[int]:
        # Prefer explicit slot
        slot = wand_cache_obj.get("current_wand_slot")
        if isinstance(slot, int) and slot in (1, 2):
            return slot
        # Match by name
        cur_name = wand_cache_obj.get("current_wand_name")
        if isinstance(cur_name, str) and cur_name:
            for i, w in enumerate(service_wands_obj, start=1):
                if isinstance(w, dict) and str(w.get("name", "")) == cur_name:
                    return i
        return None

    def print_current_wand_spells(wand_idx: int, service_wands_obj: List[Dict[str, Any]]):
        if not (1 <= wand_idx <= len(service_wands_obj)):
            print("⚠ 当前法杖索引无效，无法打印法术列表")
            return
        w = service_wands_obj[wand_idx - 1] or {}
        wand_name = w.get("name", f"法杖{wand_idx}")
        print(f"\n[当前法杖] 槽位{wand_idx}: {wand_name}")
        normal_slots = w.get("normal_slots", []) if isinstance(w.get("normal_slots", []), list) else []
        if not normal_slots:
            print("  - 无已装备法术")
            return
        for j, slot in enumerate(normal_slots, start=1):
            if slot is None:
                print(f"  - 槽位{j}: 空")
            elif isinstance(slot, dict):
                print(f"  - 槽位{j}: {slot.get('name','未知')} (ID: {slot.get('id','')})")
            else:
                print(f"  - 槽位{j}: 未知")

    current_idx = get_current_wand_index(wand_cache, wands_from_service) or 1
    print_current_wand_spells(current_idx, wands_from_service)

    def press_wand_slot(slot: int):
        # 按“第几个法杖就按几”的规则，直接用数字键位
        if 1 <= slot <= 9:
            key = str(slot)
        else:
            key = "1"
            print(f"  ⚠ 法杖{slot}超出支持范围，退回按键1")
        pydirectinput.press(key)
        time.sleep(0.3)

    def get_targets_from_cache(wand_idx_local: int) -> List[Dict[str, int]]:
        targets: List[Dict[str, int]] = []
        detected = (wand_cache.get("detected_wands") or [])
        for entry in detected:
            if not isinstance(entry, dict):
                continue
            if entry.get("wand_index") != wand_idx_local:
                continue
            slots = entry.get("slots") or {}
            normal = (slots.get("normal") or {}).get("positions") or []
            for p in normal:
                try:
                    cx, cy = int(p["x"]), int(p["y"])
                    sx, sy = client_to_screen(capturer, cx, cy)
                    targets.append({"x": sx, "y": sy})
                except Exception:
                    continue
            break
        return targets

    def get_targets_fallback_from_rows(wand_idx_local: int, override_normal_count: Optional[int] = None) -> List[Dict[str, int]]:
        try:
            rows = load_rows_from_json(SLOTS_JSON)
            if not rows or len(rows) < 2:
                print(f"  ⚠ rows数据不足: len={len(rows) if rows else 0}, 需要至少2行")
                return []
            # rows[0] = 背包；rows[1] = 法杖1, rows[2] = 法杖2, ...
            # wand_idx_local 从 1 开始，所以 wand1 -> rows[1], wand2 -> rows[2], ...
            if wand_idx_local < len(rows):
                row_idx = wand_idx_local  # wand1 -> row[1], wand2 -> row[2], wand3 -> row[3] ...
            else:
                row_idx = len(rows) - 1  # 复用最后一行
                print(f"  ⚠ 警告: 法杖{wand_idx_local}索引超出rows范围(len={len(rows)})，复用最后一行(row{row_idx})")
            if row_idx >= len(rows) or row_idx < 0:
                print(f"  ⚠ row_idx越界: {row_idx} (rows长度={len(rows)})")
                return []
            row = rows[row_idx]
            if not row:
                print(f"  ⚠ row[{row_idx}]为空，无法生成坐标")
                return []
            wands_counts, _bag_cnt = fetch_spell_counts()
            if isinstance(override_normal_count, int) and override_normal_count > 0:
                normal_cnt, post_cnt = override_normal_count, 0
            elif isinstance(wands_counts, list) and 0 <= wand_idx_local - 1 < len(wands_counts):
                w_n, w_p = wands_counts[wand_idx_local - 1]
                if w_n is not None and w_p is not None:
                    normal_cnt, post_cnt = w_n, w_p
                else:
                    normal_cnt, post_cnt = None, None
            else:
                normal_cnt, post_cnt = None, None
            if normal_cnt is None or post_cnt is None:
                # 尝试从服务端获取槽位数量
                try:
                    payload = fetch_spells_payload()
                    if isinstance(payload, dict):
                        wands_now = payload.get("Wands", []) or []
                        if 0 <= wand_idx_local - 1 < len(wands_now):
                            w = wands_now[wand_idx_local - 1]
                            if isinstance(w, dict):
                                ns = w.get("normal_slots", [])
                                if isinstance(ns, list) and len(ns) > 0:
                                    normal_cnt = len(ns)
                                    post_cnt = 0
                                else:
                                    normal_cnt = len(row) if isinstance(row, list) else 4
                                    post_cnt = 0
                            else:
                                normal_cnt = len(row) if isinstance(row, list) else 4
                                post_cnt = 0
                        else:
                            normal_cnt = len(row) if isinstance(row, list) else 4
                            post_cnt = 0
                    else:
                        normal_cnt = len(row) if isinstance(row, list) else 4
                        post_cnt = 0
                except Exception:
                    normal_cnt = len(row) if isinstance(row, list) else 4
                    post_cnt = 0
            # 如果服务端返回0或缺失，回退为当前行长度（至少保留一个可用坐标，槽位0为守护杖灵）
            if normal_cnt is None or normal_cnt <= 0:
                normal_cnt = len(row) if isinstance(row, list) else 0
                post_cnt = 0
                if normal_cnt <= 0:
                    print(f"  ⚠ normal_cnt无效且行为空，无法生成坐标")
                    return []
            normal, _post = generate_positions_from_counts(
                row=row,
                normal_count=normal_cnt,
                post_count=post_cnt,
                dx_normal=72,
                gap_normal_to_post=147,
            )
            targets: List[Dict[str, int]] = []
            for p in normal:
                sx, sy = client_to_screen(capturer, int(p["x"]), int(p["y"]))
                targets.append({"x": sx, "y": sy})
            print(f"  ℹ 法杖{wand_idx_local}目标坐标: 使用row[{row_idx}], normal_cnt={normal_cnt}, 生成{len(targets)}个坐标")
            return targets
        except Exception as e:
            print(f"  ⚠ get_targets_fallback_from_rows异常: {e}")
            import traceback
            traceback.print_exc()
            return []

    def verify_state() -> bool:
        return fetch_spells_payload() is not None

    def is_spell_enhance_or_passive(info: Dict[str, Any]) -> bool:
        if not isinstance(info, dict):
            return False
        name = str(info.get("name", "") or "")
        category = str(info.get("spell_category", "") or "")
        return ("法术增强" in name) or ("被动" in category)

    def is_magic_missile(info: Dict[str, Any]) -> bool:
        if not isinstance(info, dict):
            return False
        name = str(info.get("name", "") or "")
        return ("魔法弹" in name) or ("法术飞弹" in name) or ("Magic Missile" in name)

    def find_live_source_coord(want_name: str, current_wand_idx: int) -> Optional[Dict[str, int]]:
        """
        基于当前 /spells 状态实时查找某个法术名所在的位置，返回屏幕坐标。
        只从"背包 + 当前这根法杖"中取，不会跨法杖搬运，避免第二根法杖打乱第一根已经构筑好的布局。
        注意：守护杖灵系列法术是固定法术，不应移动，会返回 None。
        """
        # 守护杖灵系列法术不应移动
        if is_guardian_wand_spirit(want_name):
            return None
        
        payload_now = fetch_spells_payload()
        if not isinstance(payload_now, dict):
            return None
        bag_now = (payload_now.get("Bag") or [])  # type: ignore[assignment]
        wands_now = (payload_now.get("Wands") or [])  # type: ignore[assignment]

        # 调试：打印背包中的法术
        bag_spell_names = [slot.get("name", "") for slot in bag_now if isinstance(slot, dict)]
        print(f"    🔍 查找 {want_name}，背包中有: {bag_spell_names}")

        # 1) 背包：Bag 列表顺序与格子顺序一致，配合固定的 bag_targets_screen 使用
        for i, slot in enumerate(bag_now):
            if not isinstance(slot, dict):
                continue
            slot_name = slot.get("name")
            # 跳过守护杖灵系列法术
            if is_guardian_wand_spirit(slot_name):
                continue
            if slot_name == want_name and i < len(bag_targets_screen):
                print(f"    ℹ 找到来源: {want_name} 在背包槽位 {i+1} ({bag_targets_screen[i]['x']}, {bag_targets_screen[i]['y']})")
                return bag_targets_screen[i]

        # 2) 当前法杖自身：用于重排/复用当前法杖已有法术
        for w_idx, w in enumerate(wands_now, start=1):
            if w_idx != current_wand_idx:
                continue
            if not isinstance(w, dict):
                continue
            normal_slots_now = w.get("normal_slots", []) or []
            targets_now = get_targets_from_cache(w_idx)
            if not targets_now:
                targets_now = get_targets_fallback_from_rows(w_idx)
            for j, slot in enumerate(normal_slots_now):
                if not isinstance(slot, dict):
                    continue
                slot_name = slot.get("name")
                # 跳过守护杖灵系列法术（固定法术，不应移动）
                if is_guardian_wand_spirit(slot_name):
                    continue
                if slot_name == want_name and j < len(targets_now):
                    return targets_now[j]

        return None

    def sort_selection(selection: List[int]) -> List[int]:
        def sort_key(idx1based: int) -> float:
            if not (1 <= idx1based <= len(bag_spells)):
                return 1.5
            info = bag_spells[idx1based - 1].get("spell_info", {}) or {}
            if is_spell_enhance_or_passive(info):
                return 0.0
            if is_magic_missile(info):
                return 1.0
            return 2.0
        return sorted(selection, key=sort_key)

    def construct_for_wand(
        wand_idx: int,
        global_decision: Optional[Dict[str, Any]] = None,
        shared_quota: Optional[Dict[str, int]] = None,
    ):
        ws = summarize_wand_for_prompt(wand_cache, wands_from_service, wand_idx)
        n_cnt, p_cnt = get_wand_slot_counts(wand_cache, wand_idx)
        # Determine effective normal slot count (service first, then cache)
        service_n_cnt = None
        if 0 <= (wand_idx - 1) < len(wands_from_service) and isinstance(wands_from_service[wand_idx - 1], dict):
            ns = wands_from_service[wand_idx - 1].get("normal_slots", [])
            if isinstance(ns, list):
                service_n_cnt = len(ns)
        effective_n_cnt = service_n_cnt if (isinstance(service_n_cnt, int) and service_n_cnt > 0) else (n_cnt if (isinstance(n_cnt, int) and n_cnt > 0) else None)
        # Build candidate pool = bag + current wand equipped（同时也统计可用数量）
        name_to_count: Dict[str, int] = {}
        if global_decision is None:
            # 单杖模式下，本地统计可用数量（背包 + 当前法杖）
            # Count from bag（背包始终是共享资源）
            for s in bag_spells:
                info = s.get("spell_info", {}) or {}
                nm = info.get("name", "未知")
                if is_dangerous_spell_info(info, nm):
                    continue
                if is_guardian_wand_spirit(nm):
                    continue  # 跳过守护杖灵系列法术
                if nm:
                    name_to_count[nm] = name_to_count.get(nm, 0) + 1
            # 再只统计"当前这根法杖"上已装备的法术数量
            current_wand_service = wands_from_service[wand_idx - 1] if 0 <= wand_idx - 1 < len(wands_from_service) else {}
            if isinstance(current_wand_service, dict):
                eq_slots_service = current_wand_service.get("normal_slots", []) or []
                for slot_obj in eq_slots_service:
                    if isinstance(slot_obj, dict):
                        nm = slot_obj.get("name", "")
                        if is_dangerous_spell_info(None, nm):
                            continue
                        if is_guardian_wand_spirit(nm):
                            continue  # 跳过守护杖灵系列法术
                        if nm:
                            name_to_count[nm] = name_to_count.get(nm, 0) + 1
        else:
            # 全局模式下，每根法杖在开始构筑前，都基于**当前 /spells 状态**重算一遍全局可用数量
            payload_now = fetch_spells_payload() or {}
            bag_now = (payload_now.get("Bag") or [])  # type: ignore[assignment]
            wands_now = (payload_now.get("Wands") or [])  # type: ignore[assignment]
            # Bag
            for slot in bag_now:
                if not isinstance(slot, dict):
                    continue
                nm = slot.get("name")
                if not isinstance(nm, str) or not nm:
                    continue
                if is_dangerous_spell_info(None, nm):
                    continue
                if is_guardian_wand_spirit(nm):
                    continue  # 跳过守护杖灵系列法术
                name_to_count[nm] = name_to_count.get(nm, 0) + 1
            # 两根法杖当前已装备
            for w in wands_now:
                if not isinstance(w, dict):
                    continue
                eq_slots_now = w.get("normal_slots", []) or []
                for slot_obj in eq_slots_now:
                    if not isinstance(slot_obj, dict):
                        continue
                    nm = slot_obj.get("name")
                    if not isinstance(nm, str) or not nm:
                        continue
                    if is_dangerous_spell_info(None, nm):
                        continue
                    if is_guardian_wand_spirit(nm):
                        continue  # 跳过守护杖灵系列法术
                    name_to_count[nm] = name_to_count.get(nm, 0) + 1
        # 全局模式使用共享配额（跨所有法杖），否则使用本地统计
        effective_quota: Dict[str, int] = shared_quota if (global_decision is not None and isinstance(shared_quota, dict)) else name_to_count

        candidates: List[Dict[str, Any]] = []
        # bag
        for i, s in enumerate(bag_spells, start=1):
            info = s.get("spell_info", {}) or {}
            nm = info.get("name", "未知")
            if is_dangerous_spell_info(info, nm):
                continue
            if is_guardian_wand_spirit(nm):
                continue  # 跳过守护杖灵系列法术（固定法术，不应参与构筑）
            candidates.append({
                "index": i,
                "name": nm,
                "type": info.get("type", ""),
                "category": info.get("spell_category", ""),
                "attributes": info.get("attributes", ""),
                "effects": info.get("effects", []),
                "description": (info.get("all_text", "") or "")[:160],
                "available_count": name_to_count.get(nm, 0)
            })
        # equipped on this wand
        current_wand = wands_from_service[wand_idx - 1] if 0 <= wand_idx - 1 < len(wands_from_service) else {}
        eq_slots = current_wand.get("normal_slots", []) if isinstance(current_wand, dict) else []
        for slot_obj in eq_slots:
            if isinstance(slot_obj, dict):
                nm = slot_obj.get("name", "未知")
                if is_dangerous_spell_info(None, nm):
                    continue
                if is_guardian_wand_spirit(nm):
                    continue  # 跳过守护杖灵系列法术（固定法术，不应参与构筑）
                candidates.append({
                    "index": len(candidates) + 1,
                    "name": nm,
                    "type": "",
                    "category": "已装备",
                    "attributes": "",
                    "effects": [],
                    "description": f"ID: {slot_obj.get('id','')}（当前法杖已装备）",
                    "available_count": name_to_count.get(nm, 0)
                })
        if global_decision is not None:
            key = f"wand{wand_idx}"
            sub = global_decision.get(key) if isinstance(global_decision, dict) else None
            if isinstance(sub, dict):
                decision = sub
            else:
                decision = {"layout": [], "reason": "全局决策缺失"}
            print(f"\n构筑决策（法杖{wand_idx}）（来自全局决策）：")
        else:
            prompt = build_construct_prompt(ws, equipped_summary, candidates, target_count=effective_n_cnt or 4, normal_count=effective_n_cnt or n_cnt, post_count=p_cnt)
            decision = decide_with_text_model(prompt, use_think_model=True) or {"layout": [], "reason": "模型无返回"}
            print(f"\n构筑决策（法杖{wand_idx}）：")
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        # Resolve desired names layout (respect model order)
        # 支持新格式：layout 可以是对象数组 [{spell_name, slot_index}] 或字符串数组
        layout_raw = decision.get("layout", []) or []
        desired_names: List[str] = []
        slot_mapping: Dict[int, str] = {}  # slot_index -> spell_name 的映射
        
        for item in layout_raw:
            if isinstance(item, dict):
                # 新格式：对象包含 spell_name 和 slot_index
                spell_name = item.get("spell_name", "")
                slot_index = item.get("slot_index")
                if spell_name and not is_dangerous_spell_info(None, spell_name) and not is_guardian_wand_spirit(spell_name):
                    desired_names.append(spell_name)
                    if isinstance(slot_index, int) and slot_index > 0:
                        slot_mapping[slot_index] = spell_name
            elif isinstance(item, str):
                # 旧格式：字符串数组
                if not is_dangerous_spell_info(None, item) and not is_guardian_wand_spirit(item):
                    desired_names.append(item)
        
        # 过滤危险法术和守护杖灵系列法术（固定法术，不应移动）
        desired_names = [nm for nm in desired_names if not is_dangerous_spell_info(None, nm) and not is_guardian_wand_spirit(nm)]
        # 先生成目标坐标，供后续补全与上限计算使用
        targets = get_targets_fallback_from_rows(wand_idx, override_normal_count=effective_n_cnt or None)
        if not targets:
            targets = get_targets_from_cache(wand_idx)
        if not targets:
            print("⚠ 无可用目标坐标，跳过")
            return
        # 如果模型未给出布局，进行简易回填（使用可用法术，优先增益再输出）
        if not desired_names:
            print("ℹ 模型未给出布局，使用本地回填逻辑")
            # 先构建一个本地可用法术列表（扣除守护杖灵）
            available_pool: List[str] = []
            # bag
            for s in bag_spells:
                info = s.get("spell_info", {}) or {}
                nm = info.get("name", "")
                if not nm or is_dangerous_spell_info(info, nm) or is_guardian_wand_spirit(nm):
                    continue
                if effective_quota.get(nm, 0) > 0:
                    available_pool.append(nm)
            # 当前法杖已有
            current_wand_slots = current_wand.get("normal_slots", []) if isinstance(current_wand, dict) else []
            for slot_obj in current_wand_slots:
                if isinstance(slot_obj, dict):
                    nm = slot_obj.get("name", "")
                    if nm and not is_dangerous_spell_info(None, nm) and not is_guardian_wand_spirit(nm):
                        available_pool.append(nm)
            # 去重但保留顺序
            seen: set[str] = set()
            pool_unique = []
            for nm in available_pool:
                if nm not in seen:
                    pool_unique.append(nm)
                    seen.add(nm)
            # 排序：增益优先，再法术飞弹，再其他
            def local_priority(nm: str) -> int:
                enhance_kw = ["法术增强", "伤害强化", "分裂", "多重射击", "闪电链", "奥术新星", "时长强化", "范围增强", "冷却", "公转", "反弹", "穿透"]
                projectile_kw = ["法术飞弹", "魔法弹", "Magic Missile", "蝴蝶", "彩虹", "激光", "落雷", "黑洞", "冥蛇", "滚石", "诡雷", "瓦解射线", "注魔硬币"]
                if any(k in nm for k in enhance_kw):
                    return 0
                if any(k in nm for k in projectile_kw):
                    return 1
                return 2
            pool_sorted = sorted(pool_unique, key=local_priority)
            desired_names = pool_sorted[: max(0, min(len(targets), effective_n_cnt or len(targets)))]
        # If model returned fewer than available normal slots, backfill using simple heuristic（仅单杖模式启用）
        need_count = 0
        if global_decision is None:
            if isinstance(effective_n_cnt, int) and effective_n_cnt > 0:
                need_count = max(0, min(effective_n_cnt, len(targets)) - len(desired_names))
            else:
                need_count = max(0, len(targets) - len(desired_names))
            if need_count > 0:
                existing = set(desired_names)
                # Check if model's layout includes summon-related spells
                summon_keywords = ["魔导书", "啵啵", "杖灵", "召唤"]
                has_summon_in_layout = any(any(kw in nm for kw in summon_keywords) for nm in desired_names)
                # candidate name list from service (bag + both wands), keep order
                candidate_names: List[str] = []
                for slot in bag_from_service:
                    if isinstance(slot, dict):
                        nm = slot.get("name")
                        if isinstance(nm, str):
                            if is_dangerous_spell_info(None, nm):
                                continue
                            candidate_names.append(nm)
                for widx2, w2 in enumerate(wands_from_service, start=1):
                    if isinstance(w2, dict):
                        for s in (w2.get("normal_slots", []) or []):
                            if isinstance(s, dict):
                                nm = s.get("name")
                                if isinstance(nm, str):
                                    if is_dangerous_spell_info(None, nm):
                                        continue
                                    candidate_names.append(nm)
                # heuristic priority: passives/enhances first, then utility, then others
                # Exclude summon-specific enhancers if no summon spells in layout
                def name_priority(nm: str) -> int:
                    summon_only_keywords = ["巨魔血清", "脐带", "寄生虫"]
                    if not has_summon_in_layout and any(kw in nm for kw in summon_only_keywords):
                        return 999  # Exclude these
                    passive_keywords = ["法术增强", "伤害强化", "多重射击", "分裂", "闪电链", "奥术新星", "时长强化", "范围增强", "冷却", "公转", "反弹", "穿透"]
                    utility_keywords = ["拟态", "储魔", "法术汲取", "共鸣", "杖灵", "自动导航", "寻踪"]
                    if any(k in nm for k in passive_keywords):
                        return 0
                    if any(k in nm for k in utility_keywords):
                        return 1
                    return 2
                sorted_candidates = sorted([n for n in candidate_names if n not in existing], key=name_priority)
                for nm in sorted_candidates:
                    priority_val = name_priority(nm)
                    if priority_val >= 999:  # Excluded
                        continue
                    desired_names.append(nm)
                    existing.add(nm)
                    need_count -= 1
                    if need_count <= 0:
                        break
        # Enforce passive/enhance left alignment finally (stable)
        def final_priority(nm: str) -> int:
            passive_keywords = ["法术增强", "伤害强化", "多重射击", "分裂", "闪电链", "奥术新星", "时长强化", "范围增强", "冷却", "公转", "反弹", "穿透", "拟态", "储魔", "法术汲取", "共鸣", "杖灵", "自动导航", "寻踪"]
            return 0 if any(k in nm for k in passive_keywords) else 1
        desired_names = sorted(desired_names, key=final_priority)
        print(f"目标布局（左→右）: {desired_names}")
        # 严格限制在 normal 槽数量内，避免拖拽到 post 区域
        # 优先使用服务端当前法杖 normal 槽数量作为上限，其次缓存n_cnt，最后用targets长度
        # effective_n_cnt 已在前面计算
        # 计算最大可放置数量（槽位1固定守护杖灵，所以可用槽位是 effective_n_cnt - 1）
        if isinstance(effective_n_cnt, int) and effective_n_cnt > 0:
            # 槽位1固定守护杖灵，所以可用槽位是 effective_n_cnt - 1
            available_slots = effective_n_cnt - 1
            max_slots = min(len(desired_names), available_slots, len(targets) - 1)  # targets 也是0-based，槽位0是守护杖灵
        else:
            # 如果没有 effective_n_cnt，假设槽位1是守护杖灵，可用槽位是 len(targets) - 1
            available_slots = len(targets) - 1 if len(targets) > 1 else 0
            max_slots = min(len(desired_names), available_slots)
        if len(desired_names) > max_slots:
            print(f"ℹ 规划 {len(desired_names)} 个，但受可用normal槽 {available_slots} 限制，将放置 {max_slots} 个")

        # 使用 name_to_count 作为来源配额（背包 + 两根法杖），不依赖静态 source_pool
        available_quota: Dict[str, int] = dict(effective_quota)
        print(f"来源计数: {available_quota}")
        # Check if model's layout includes summon-related spells to determine if summon-specific enhancers are relevant
        summon_keywords = ["魔导书", "啵啵", "杖灵", "召唤"]
        has_summon_in_layout = any(any(kw in nm for kw in summon_keywords) for nm in desired_names)
        # Reduce desired_names by quota; collect deficit to backfill
        final_names: List[str] = []
        replaced_log: List[str] = []
        for nm in desired_names:
            if available_quota.get(nm, 0) > 0:
                final_names.append(nm)
                available_quota[nm] -= 1
            else:
                replaced_log.append(f"{nm}（未找到来源）")
        # backfill if still short
        def name_priority_fill(nm: str) -> int:
            # Exclude summon-specific enhancers if no summon spells in layout
            summon_only_keywords = ["巨魔血清", "脐带", "寄生虫"]
            if not has_summon_in_layout and any(kw in nm for kw in summon_only_keywords):
                return 999  # Exclude these
            passive_keywords = ["法术增强", "伤害强化", "多重射击", "分裂", "闪电链", "奥术新星", "时长强化", "范围增强", "冷却", "公转", "反弹", "穿透", "拟态", "储魔", "法术汲取", "共鸣", "自动导航", "寻踪"]
            if any(k in nm for k in passive_keywords):
                return 0
            utility_keywords = ["拟态", "储魔", "法术汲取", "共鸣", "自动导航", "寻踪"]
            if any(k in nm for k in utility_keywords):
                return 1
            return 2
        if len(final_names) < max_slots:
            # prepare candidate list expanded by quota, excluding irrelevant ones
            expanded: List[str] = []
            for nm, cnt in available_quota.items():
                if cnt > 0:
                    if is_dangerous_spell_info(None, nm):
                        continue
                    if is_guardian_wand_spirit(nm):
                        continue  # 跳过守护杖灵系列法术
                    expanded.extend([nm] * cnt)
            expanded_sorted = sorted(expanded, key=name_priority_fill)
            for nm in expanded_sorted:
                if len(final_names) >= max_slots:
                    break
                priority_val = name_priority_fill(nm)
                if priority_val >= 999:  # Excluded
                    continue
                final_names.append(nm)
                available_quota[nm] = available_quota.get(nm, 0) - 1
                if nm not in desired_names:
                    replaced_log.append(f"{nm}（补位）")
        # cap to max_slots
        final_names = final_names[:max_slots]
        # Re-sort final_names to ensure enhance spells are left of projectile spells, projectiles push to rightmost
        def final_priority_resort(nm: str) -> int:
            """排序优先级：0=法术增强/被动（最左），1=其他，2=法术飞弹（最右）"""
            # 法术增强类（必须在法术飞弹左侧）
            enhance_keywords = ["法术增强", "伤害强化", "多重射击", "分裂", "闪电链", "奥术新星", "时长强化", "范围增强", "冷却", "公转", "反弹", "穿透"]
            # 其他被动/辅助
            other_passive_keywords = ["拟态", "储魔", "法术汲取", "共鸣", "自动导航", "寻踪"]
            # 法术飞弹类（必须在增强右侧）
            projectile_keywords = ["法术飞弹", "魔法弹", "Magic Missile", "蝴蝶", "彩虹", "激光", "落雷", "黑洞", "冥蛇", "滚石", "诡雷", "瓦解射线", "注魔硬币"]
            
            if any(k in nm for k in enhance_keywords):
                return 0  # 法术增强最左
            elif any(k in nm for k in other_passive_keywords):
                return 1  # 其他被动次左
            elif any(k in nm for k in projectile_keywords):
                return 2  # 法术飞弹最右
            else:
                return 1  # 其他法术（可能是召唤等）
        
        # 执行排序：确保增强在飞弹左侧
        final_names = sorted(final_names, key=final_priority_resort)
        # 确保至少有一个法术飞弹类法术，且放在最右侧
        has_projectile = any(is_projectile_spell(nm) for nm in final_names)
        if not has_projectile:
            # 尝试从剩余可用配额中添加一个法术飞弹
            for nm, cnt in available_quota.items():
                if cnt > 0 and is_projectile_spell(nm) and nm not in final_names:
                    final_names.append(nm)
                    available_quota[nm] = available_quota.get(nm, 0) - 1
                    break
        # 把所有法术飞弹类移动到最右侧（相对顺序保持）
        projectiles = [nm for nm in final_names if is_projectile_spell(nm)]
        non_projectiles = [nm for nm in final_names if not is_projectile_spell(nm)]
        final_names = non_projectiles + projectiles
        if replaced_log:
            print(f"ℹ 替换日志: {', '.join(replaced_log)}")
        print(f"最终布局（应用配额后，左→右）: {final_names}")

        # Verify and drag
        press_wand_slot(wand_idx)
        if not verify_state():
            print("⚠ 接口校验失败（构筑前）")
        
        # 获取当前法杖的已装备法术，用于检查哪些槽位有守护杖灵
        def get_current_wand_slots() -> List[Dict[str, Any]]:
            payload_now = fetch_spells_payload()
            if not isinstance(payload_now, dict):
                return []
            wands_now = (payload_now.get("Wands") or [])  # type: ignore[assignment]
            if 0 <= wand_idx - 1 < len(wands_now):
                w = wands_now[wand_idx - 1]
                if isinstance(w, dict):
                    return w.get("normal_slots", []) or []
            return []
        
        skipped_missing_sources: List[str] = []
        placed_names: List[str] = []
        placed_target_idx = 1  # 槽位1固定守护杖灵，从槽位2开始放置（0-based index=1）
        # 如果模型返回了槽位映射，使用它；否则使用自动分配
        use_slot_mapping = len(slot_mapping) > 0
        
        # 创建反向映射：spell_name -> slot_index
        spell_to_slot: Dict[str, int] = {}
        for slot_idx, spell_nm in slot_mapping.items():
            spell_to_slot[spell_nm] = slot_idx
        
        for i in range(len(final_names)):
            want_name = final_names[i]
            # 跳过守护杖灵系列法术（固定法术，不应移动）
            if is_guardian_wand_spirit(want_name):
                print(f"  ℹ 跳过守护杖灵系列法术: {want_name}（固定法术，不应移动）")
                skipped_missing_sources.append(want_name)
                continue
            # 每次拖拽前实时从当前 /spells 状态确定来源坐标，避免使用过期的格子坐标
            src = find_live_source_coord(want_name, wand_idx)
            if src is None:
                print(f"  ⚠ 未找到法术来源: {want_name}，跳过该位")
                skipped_missing_sources.append(want_name)
                continue
            
            # 确定目标槽位
            target_slot_idx = None
            if use_slot_mapping and want_name in spell_to_slot:
                # 使用模型返回的槽位位置（转换为0-based索引）
                model_slot = spell_to_slot[want_name] - 1
                if model_slot == 0:
                    print(f"  ⚠ 模型槽位为1（守护杖灵固定），跳过自动分配")
                elif 0 <= model_slot < len(targets):
                    target_slot_idx = model_slot
                    print(f"  ℹ 使用模型指定的槽位: {spell_to_slot[want_name]}")
                else:
                    print(f"  ⚠ 模型返回的槽位位置无效: {spell_to_slot[want_name]}，使用自动分配")
            
            if target_slot_idx is None:
                # 自动分配：查找下一个可用的目标槽位（跳过槽位0的守护杖灵）
                current_slots = get_current_wand_slots()
                # 确保不超过已放置的法术数量限制（max_slots 表示可用槽位数量，不包括槽位0）
                while placed_target_idx < len(targets):
                    if placed_target_idx == 0:
                        # 槽位0固定是守护杖灵，跳过
                        placed_target_idx += 1
                        continue
                    # 检查是否已经放置了足够多的法术（max_slots 不包括槽位0）
                    if len(placed_names) >= max_slots:
                        break
                    # 检查目标槽位是否已有守护杖灵
                    if placed_target_idx < len(current_slots):
                        slot_obj = current_slots[placed_target_idx]
                        if isinstance(slot_obj, dict):
                            slot_name = slot_obj.get("name", "")
                            if is_guardian_wand_spirit(slot_name):
                                print(f"  ℹ 跳过槽位{placed_target_idx + 1}（已有守护杖灵: {slot_name}）")
                                placed_target_idx += 1
                                continue
                    target_slot_idx = placed_target_idx
                    break
                if target_slot_idx is None:
                    # 如果已经超过 targets 长度，退一步使用最后一个可用槽位（非守护杖灵）
                    if len(targets) > 1 and placed_target_idx >= len(targets):
                        fallback_idx = len(targets) - 1
                        # 避免槽位0
                        if fallback_idx == 0 and len(targets) > 1:
                            fallback_idx = 1
                        target_slot_idx = fallback_idx
                        print(f"  ⚠ 目标槽位索引超界，改用最后一个槽位 {target_slot_idx + 1}")
                    else:
                        print(f"  ⚠ 没有可用目标槽位，跳过 {want_name} (已放置 {len(placed_names)}/{max_slots})")
                        skipped_missing_sources.append(want_name)
                        continue
            
            if target_slot_idx >= len(targets):
                print(f"  ⚠ 目标槽位超出范围: {target_slot_idx + 1}，跳过 {want_name}")
                skipped_missing_sources.append(want_name)
                continue
            
            dst = targets[target_slot_idx]
            # 更新 placed_target_idx（用于下次自动分配）
            if not use_slot_mapping:
                placed_target_idx = target_slot_idx + 1
            # 验证坐标有效性
            src_x, src_y = int(src.get("x", 0)), int(src.get("y", 0))
            dst_x, dst_y = int(dst.get("x", 0)), int(dst.get("y", 0))
            if src_x <= 0 or src_y <= 0 or dst_x <= 0 or dst_y <= 0:
                print(f"  ⚠ 坐标无效: 来源({src_x}, {src_y}) -> 目标({dst_x}, {dst_y})，跳过")
                skipped_missing_sources.append(want_name)
                continue
            # 检查来源和目标是否是同一个位置（避免无意义的拖拽）
            if abs(src_x - dst_x) < 10 and abs(src_y - dst_y) < 10:
                print(f"  ℹ 跳过：{want_name} 已在目标位置 ({src_x}, {src_y})")
                placed_target_idx += 1  # 该槽位已经有正确的法术，跳过
                continue
            # Before each drag, re-verify and ensure wand hotkey
            press_wand_slot(wand_idx)
            if not verify_state():
                print("  ⚠ 接口校验失败（拖拽前），重试继续")
            try:
                print(f"  → 拖拽 {want_name}: ({src_x}, {src_y}) -> ({dst_x}, {dst_y})")
                # 确保游戏窗口有焦点（点击窗口中心区域）
                try:
                    window_info = capturer.get_window_info()
                    center_x = window_info.get("left", 0) + window_info.get("width", 0) // 2
                    center_y = window_info.get("top", 0) + window_info.get("height", 0) // 2
                    pydirectinput.click(center_x, center_y)
                    time.sleep(0.1)
                except Exception:
                    pass
                # 使用 pydirectinput 进行更可靠的鼠标操作
                pydirectinput.moveTo(src_x, src_y)
                time.sleep(0.15)
                pydirectinput.mouseDown(button='left')
                time.sleep(0.1)
                # pydirectinput 的 moveTo 不支持 duration，使用多步移动模拟平滑移动
                steps = 5
                for step in range(1, steps + 1):
                    t = step / steps
                    interp_x = int(src_x + (dst_x - src_x) * t)
                    interp_y = int(src_y + (dst_y - src_y) * t)
                    pydirectinput.moveTo(interp_x, interp_y)
                    time.sleep(0.04)
                time.sleep(0.1)
                pydirectinput.mouseUp(button='left')
                time.sleep(0.4)
                placed_names.append(want_name)
                placed_target_idx += 1
                # 成功放置后扣减全局/本地可用配额，避免重复使用
                if want_name in effective_quota and effective_quota[want_name] > 0:
                    effective_quota[want_name] -= 1
            except Exception as e:
                print(f"  ✗ 拖拽失败: {e}")
                continue
            press_wand_slot(wand_idx)
            if not verify_state():
                print("  ⚠ 接口校验失败（拖拽后），继续下一步")
        if skipped_missing_sources:
            print(f"ℹ 本轮未能放置（未找到来源）: {skipped_missing_sources}")
        print(f"✓ 已放置 {len(placed_names)}/{max_slots}: {placed_names}")

    # Construct only for available wands
    wand_count = sum(1 for w in wands_from_service if isinstance(w, dict))
    if wand_count == 0:
        print("\n⚠ 未检测到任何法杖，结束构筑流程")
        return

    # 全局一次性规划两根法杖的布局
    global_prompt_info = build_global_construct_prompt(
        wand_cache=wand_cache,
        wands_from_service=wands_from_service,
        bag_spells=bag_spells,
        equipped_summary=equipped_summary,
    )
    global_prompt = global_prompt_info.get("prompt", "")
    print(f"global prompt:{global_prompt_info}")
    wand_normals = global_prompt_info.get("wand_normals", {}) or {}
    global_name_to_count = global_prompt_info.get("global_name_to_count", {}) or {}

    global_decision = decide_with_text_model(global_prompt, use_think_model=True) or {}
    # 基于全局配额做一次严格裁剪，确保 wand1+wand2 使用总量不超过 available_count
    if isinstance(global_decision, dict):
        global_decision = sanitize_global_decision(global_decision, global_name_to_count, wand_normals)
    print("\n全局构筑决策：")
    try:
        print(json.dumps(global_decision, ensure_ascii=False, indent=2))
    except Exception:
        print(global_decision)

    shared_available = dict(global_name_to_count)
    for wand_idx in range(1, wand_count + 1):
        construct_for_wand(wand_idx, global_decision=global_decision, shared_quota=shared_available)

    print("\n✓ 构筑流程完成（独立测试）")


if __name__ == "__main__":
    run()


