import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import urllib.request
import urllib.error

from omni_models.omni import get_text_client, get_text_model
from store_purchase_decision import StorePurchaseDecision
# from analyze_spells import SpellAnalyzer  # 已删除，如需使用请重新实现

import pydirectinput
import time as _time
from get_game_window import GameWindowCapture
from mark_spell_slots import load_rows_from_json, generate_positions_from_counts, fetch_spell_counts

# 与 spell_construct_flow 保持一致的危险/关键法术定义
DANGEROUS_SPELL_NAMES = {"诡雷"}
DANGEROUS_KEYWORDS = ("无差别伤害", "自伤", "反弹", "爆炸")
PROJECTILE_KEYWORDS = ["法术飞弹", "魔法弹", "Magic Missile", "蝴蝶", "彩虹", "激光", "落雷", "黑洞", "冥蛇", "滚石", "诡雷", "瓦解射线", "注魔硬币", "审判之剑", "次元行者"]


import os
from pathlib import Path
from dotenv import load_dotenv

# 加载.env文件
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

SPELLS_ENDPOINT = os.getenv('SPELLS_ENDPOINT', 'http://localhost:1234/spells')
WAND_CACHE_FILE = "store_wand_cache.json"


def fetch_spells_payload(endpoint: str = SPELLS_ENDPOINT) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(endpoint, timeout=1.5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError):
        return None


def is_dangerous_spell_info(info: Optional[Dict[str, Any]], fallback_name: str = "") -> bool:
    """
    与 spell_construct_flow 对齐：检测潜在自伤或危险法术，避免在构筑与购买后使用。
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


def is_guardian_wand_spirit(spell_name: str) -> bool:
    """守护杖灵系列是固定法术，不应移动或参与构筑。"""
    if not spell_name:
        return False
    name = str(spell_name).strip()
    return name in ("守护杖灵", "守护杖灵+", "守护杖灵++")


def is_projectile_spell(name: str) -> bool:
    if not name:
        return False
    return any(kw in name for kw in PROJECTILE_KEYWORDS)


def load_wand_cache(path: Path = Path(WAND_CACHE_FILE)) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def summarize_wands_for_prompt(wand_cache: Dict[str, Any], service_wands: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    # From service (basic guarantees)
    if isinstance(service_wands, list):
        for i, w in enumerate(service_wands):
            if not isinstance(w, dict):
                continue
            name = w.get("name", f"法杖{i+1}")
            lines.append(f"- {name}: max_mp={w.get('max_mp')}, mp_recover={w.get('mp_recover')}, shoot_interval={w.get('shoot_interval')}, cooldown={w.get('cooldown')}")
    # From cache (panel_info might include extra special descriptions)
    detected = wand_cache.get("detected_wands", [])
    if isinstance(detected, list):
        for entry in detected:
            if not isinstance(entry, dict):
                continue
            name = entry.get("item_name") or (entry.get("basic", {}) or {}).get("name") or "未知法杖"
            panel = entry.get("panel_info") or {}
            special = panel.get("description") or panel.get("attributes") or ""
            special = special.strip()
            if special:
                lines.append(f"- {name} 额外描述: {special}")
    return "\n".join(lines)


def summarize_equipped_spells_for_prompt(service_wands: List[Dict[str, Any]]) -> str:
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


def build_purchase_prompt(coin: int, items: List[Dict[str, Any]], wand_summary: str, equipped_summary: str) -> str:
    items_text = []
    items_with_index = []
    for idx, item in enumerate(items):
        item_id = item.get('id')
        item_name = item.get('name', '')
        item_type = item.get('type', -1)
        item_price = item.get('price', 0)
        item_info = item.get('info')
        desc = f"- {item_name} (ID: {item_id}, 价格: {item_price}金币"
        if item_type == 0:
            desc += ", 类型: 法杖"
        elif item_type == 1:
            desc += ", 类型: 法术"
        elif item_type == 4:
            desc += ", 类型: 消耗品"
        if item_info:
            if item_type == 0:
                desc += f", 属性: {item_info.get('attributes', '')}, 效果: {item_info.get('effects', [])}"
            elif item_type == 1:
                desc += f", 类别: {item_info.get('spell_category', '')}, 伤害: {item_info.get('damage', '')}, 效果: {item_info.get('effects', [])}"
        desc += ")"
        items_text.append(desc)
        items_with_index.append(f"索引{idx}: {item_name} - {item_price}金币")
    items_summary = "\n".join(items_text)
    items_index_summary = "\n".join(items_with_index)

    prompt = f"""以下是游戏商店中的商品信息，请你帮助判断应该购买哪些商品，并给出简短理由。

当前法杖（含基础与特殊描述）：
{wand_summary if wand_summary else "无"}

当前已装备法术：
{equipped_summary if equipped_summary else "无"}

重要约束：购买商品的总价绝对不能超过当前金币 {coin}！

当前金币: {coin}

商品列表（带索引和价格）：
{items_index_summary}

商品详细信息：
{items_summary}

商品类型说明：
- type 0 = 法杖：主要武器，影响攻击方式和伤害
- type 1 = 法术：主动或被动技能，提供各种效果
- type 4 = 消耗品：一次性使用的道具（如钥匙、护盾等）

购买策略建议：
1. 必须遵守：所选商品的总价不能超过当前金币 {coin}
2. 优先考虑能提升战斗力的核心装备（法杖、核心法术）
3. 结合当前法杖基础与特殊描述、已装备法术情况做出最佳选择
4. 法杖和法术要考虑与当前构筑的配合度
5. 避免购买可能自伤/反弹/爆炸的危险法术（例如诡雷），若描述出现“无差别伤害/自伤/反弹/爆炸”等关键词则不要购买。

只允许输出以下JSON格式，不得添加额外内容：
{{"purchases": [0, 2, 5], "reason": "购买理由"}}
请确保purchases中所有商品的价格总和不超过 {coin} 金币！
如果不需要购买任何商品，返回空数组：{{"purchases": [], "reason": "不购买的理由"}}"""
    return prompt


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


def decide_with_text_model(prompt: str) -> Optional[Dict[str, Any]]:
    client = get_text_client()
    model = get_text_model()
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
        reply = completion.choices[0].message.content
    except Exception as exc:
        print(f"✗ 调用文本模型失败: {exc}")
        return None
    decision = parse_json(reply)
    if decision is None:
        print("✗ 无法解析文本模型的JSON回复")
        print(f"原始回复: {reply}")
        return None
    return decision


def build_construct_prompt(
    wand_summary: str,
    equipped_summary: str,
    bag_spells_summary: List[Dict[str, Any]],
    target_count: int = 4,
    normal_count: Optional[int] = None,
    post_count: Optional[int] = None,
) -> str:
    summary_text = json.dumps(bag_spells_summary, ensure_ascii=False, indent=2)
    constraints = []
    if normal_count is not None:
        constraints.append(f"- 可装备的 normal 槽位数量: {normal_count}")
    if post_count is not None:
        constraints.append(f"- 法杖自带的 post 槽位数量: {post_count}（自动触发，不与 normal 槽互动）")
    slots_constraints = "\n".join(constraints) if constraints else "（槽位数量未知）"

    prompt = f"""请基于当前法杖与已装备法术的描述，从背包法术中选择最优的{target_count}个法术用于构筑，并说明理由。

当前法杖（含基础与特殊描述）：
{wand_summary if wand_summary else "无"}

当前已装备法术：
{equipped_summary if equipped_summary else "无"}

槽位约束：
{slots_constraints}

背包法术（解析后的摘要，index从1开始）：
{summary_text}

规则与建议：
1. 避免危险法术（诡雷/自伤/反弹/爆炸等关键字）。
2. normal 槽与 post 槽分开，post 槽是“法杖自触发”，不是拖拽目标。
3. 相对位置必须满足“被动/增强在左，主动输出在右”，法术增强（伤害强化/分裂/多重射击等）必须放在法术飞弹类（魔法弹/法术飞弹/蝴蝶/彩虹等）左侧才生效。
4. 在可用 normal 槽内，必须包含至少一个核心输出，优先法术飞弹类；被动不要占满所有槽位。
5. 结合当前法杖属性（射速/冷却/充能等）选择稳定命中的持续输出。
6. 至少保证一个法术飞弹类输出位于最右侧的可用槽位。
7. 若候选不足，可重复同名，但出现次数不要超过它在背包中的数量。

只返回JSON：{{"selected": [1, 3, 5, 7], "reason": "一句话理由"}}"""
    return prompt


def run():
    print("=" * 60)
    print("商店自动购买 + 法术构筑（整合版）")
    print("=" * 60)



    wand_cache = load_wand_cache()
    service_payload = fetch_spells_payload() or {}
    wands_from_service = service_payload.get("Wands", []) or []

    wand_summary = summarize_wands_for_prompt(wand_cache, wands_from_service)
    equipped_summary = summarize_equipped_spells_for_prompt(wands_from_service)

    # 1) 准备购买流程（但用增强后的提示词做决策）
    decider = StorePurchaseDecision()

    print("\n[步骤1] 获取商店数据...")
    store_data = decider.fetch_store_data()
    if not store_data:
        print("✗ 无法获取商店数据")
        return

    print("\n[步骤2] 获取当前金币...")
    coin = decider.get_current_coin()

    print("\n[步骤3] 分析商品信息...")
    analyzed_items = decider.analyze_all_items(store_data)

    print("\n[步骤4] 用增强提示词进行购买决策...")
    purchase_prompt = build_purchase_prompt(coin, analyzed_items, wand_summary, equipped_summary)
    decision = decide_with_text_model(purchase_prompt) or {"purchases": [], "reason": "模型无返回"}

    # 验证预算
    purchases = decision.get("purchases", [])
    total_price = 0
    for idx in purchases:
        if 0 <= idx < len(analyzed_items):
            total_price += analyzed_items[idx].get("price", 0)
    if total_price > coin:
        print(f"⚠ 建议总价 {total_price} 超出当前金币 {coin}，进行修正...")
        affordable = []
        remaining = coin
        for idx in purchases:
            if 0 <= idx < len(analyzed_items):
                price = analyzed_items[idx].get("price", 0)
                if price <= remaining:
                    affordable.append(idx)
                    remaining -= price
        decision["purchases"] = affordable

    print("\n决策结果:")
    print(json.dumps(decision, ensure_ascii=False, indent=2))

    # 5) 执行购买（支持在有两根法杖时先决定替换哪根，并在法杖购买前切换到对应法杖）
    def ask_wand_replacement(service_wands: List[Dict[str, Any]]) -> int:
        # 返回 1 或 2，默认 1
        lines = []
        for i, w in enumerate(service_wands, start=1):
            if not isinstance(w, dict):
                continue
            lines.append(f"{i}: {w.get('name','未知')} (max_mp={w.get('max_mp')}, mp_recover={w.get('mp_recover')}, shoot_interval={w.get('shoot_interval')}, cooldown={w.get('cooldown')})")
        summary = "\n".join(lines) if lines else "未知"
        prompt = f"""当前已经拥有两根法杖，请选择替换哪一根（按1或2选择），只返回JSON：
{summary}
格式：{{"replace": 1}} 或 {{"replace": 2}}"""
        dec = decide_with_text_model(prompt) or {}
        rep = int(dec.get("replace", 1))
        return 1 if rep != 2 else 2

    def press_wand_slot(slot: int):
        # 聚焦窗口的逻辑在下层类中已处理/或使用系统焦点
        key = "1" if slot == 1 else "2"
        pydirectinput.press(key)
        _time.sleep(0.3)

    def execute_purchases_itemwise(purchase_indices: List[int], items: List[Dict[str, Any]], current_coin: int):
        print(f"\n开始执行购买 {len(purchase_indices)} 个商品...")
        store_data_live = decider.fetch_store_data()
        if not store_data_live:
            print("✗ 无法获取商店数据，无法执行购买")
            return
        store_items = store_data_live.get('store', [])

        # 判断是否存在法杖购买
        has_wand_purchase = any((0 <= idx < len(items)) and items[idx].get("type") == 0 for idx in purchase_indices)
        # 是否已有两根法杖
        replacement_choice = None
        payload_now = fetch_spells_payload() or {}
        wands_now = [w for w in (payload_now.get("Wands") or []) if isinstance(w, dict)]
        if has_wand_purchase and len(wands_now) >= 2:
            replacement_choice = ask_wand_replacement(wands_now)
            print(f"  ✓ 将替换法杖: {replacement_choice}")

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
            item_type = item.get('type', -1)
            item_info = item.get('info', {}) or {}

            # 跳过危险法术的购买
            if item_type == 1 and is_dangerous_spell_info(item_info, item_name):
                print(f"  ⚠ 跳过危险法术购买: {item_name}")
                continue

            # 切换到要替换的法杖（仅在本次要购买的是法杖且已有两根时）
            if item_type == 0 and replacement_choice in (1, 2):
                print(f"  [预切换] 准备替换法杖{replacement_choice}，按下 {replacement_choice}")
                press_wand_slot(replacement_choice)

            # 检查金币是否足够
            if total_spent + item_price > current_coin:
                print(f"  ⚠ 金币不足，无法购买 {item_name} (需要 {item_price}，已花费 {total_spent}，剩余 {current_coin - total_spent})")
                continue

            # 找到货架位置
            shelf_index = None
            for i, store_item in enumerate(store_items):
                if store_item.get('id') == item_id:
                    shelf_index = i
                    break
            if shelf_index is None:
                print(f"  ✗ 未找到商品 {item_name} 的货架位置")
                continue

            print(f"\n[购买商品 {purchased_count + 1}/{len(purchase_indices)}] {item_name} (价格: {item_price}金币)")
            print(f"  [步骤1] 移动到货架 {shelf_index + 1}...")
            if not decider.move_to_shelf(shelf_index):
                print(f"  ✗ 移动到货架失败，跳过购买 {item_name}")
                continue

            # 按E键购买
            print(f"  [步骤2] 按E键购买...")
            try:
                pydirectinput.press('e')
                _time.sleep(0.4)
                print(f"  ✓ 已按E键购买 {item_name}")
                purchased_count += 1
                total_spent += item_price
                current_coin -= item_price
            except Exception as e:
                print(f"  ✗ 按E键失败: {e}")
                continue

        print(f"\n✓ 购买完成: 成功购买 {purchased_count} 个商品，总花费 {total_spent} 金币")

    if decision.get("purchases"):
        print("\n[步骤5] 执行购买（逐项）...")
        execute_purchases_itemwise(decision["purchases"], analyzed_items, coin)
    else:
        print("\n无购买项，跳过购买执行")

    # 6) 法术分析（获取背包法术解析结果）
    print("\n[步骤6] 分析背包与已装备法术...")
    # TODO: SpellAnalyzer已删除，需要重新实现或使用替代方案
    print("⚠ SpellAnalyzer模块已删除，此功能需要重新实现")
    return
    # 以下代码需要重新实现
    # sa = SpellAnalyzer()
    # if not sa.load_coordinates():
    #     print("✗ 无法加载法术槽坐标，跳过构筑")
    #     return
    # if not sa.capturer.select_magicraft_window():
    #     print("✗ 无法找到Magicraft游戏窗口，跳过构筑")
    #     return
    # 获取接口数据并判断哪些槽位有法术
    count_result, _ = sa.analyze_spell_count()
    if count_result is None:
        print("✗ 无法判断法术槽")
        return
    slots_indices = sa.parse_spell_slots(count_result)

    # 分析背包中的法术详情（仅有法术的槽位）
    bag_spells: List[Dict[str, Any]] = []
    for slot_index in slots_indices:
        # 从接口数据拿ID与名称，传给分析以使用缓存
        spell_id = None
        spell_name = ""
        if slot_index < len(sa.latest_bag_slots):
            slot_data = sa.latest_bag_slots[slot_index]
            if isinstance(slot_data, dict):
                spell_id = slot_data.get("id")
                spell_name = slot_data.get("name", "")
        desc = sa.analyze_spell_description(slot_index, spell_id=spell_id, spell_name=spell_name)
        if desc:
            spell_info = sa.parse_spell_info(desc)
            if spell_info:
                sa.cache_spell_info(spell_id=spell_id, spell_name=spell_name, spell_info=spell_info)
                # 记录来源坐标以便拖拽
                if is_dangerous_spell_info(spell_info, spell_name) or is_guardian_wand_spirit(spell_name):
                    # 危险法术或守护杖灵直接跳过，不进入构筑候选
                    continue
                coord = sa.coordinates[slot_index] if slot_index < len(sa.coordinates) else {}
                bag_spells.append({"index": len(bag_spells) + 1, "spell_info": spell_info, "coordinate": coord})
        time.sleep(0.3)
    sa.save_spell_cache()

    # 7) 用增强提示词分别为两个法杖进行法术构筑决策，并执行拖拽
    print("\n[步骤7] 分别为两个法杖选择构筑并拖拽（每次操作前查询接口校验）...")
    bag_summary_for_prompt: List[Dict[str, Any]] = []
    for i, s in enumerate(bag_spells, start=1):
        info = s.get("spell_info", {}) or {}
        bag_summary_for_prompt.append({
            "index": i,
            "name": info.get("name", "未知"),
            "type": info.get("type", ""),
            "category": info.get("spell_category", ""),
            "attributes": info.get("attributes", ""),
            "effects": info.get("effects", []),
            "description": (info.get("all_text", "") or "")[:160]
        })

    # 构造每个法杖的提示（分别基于该法杖的摘要）
    def get_wand_specific_summary(wand_idx: int) -> str:
        # 从服务端与缓存中抽取对应法杖的信息
        lines = []
        if 0 <= wand_idx - 1 < len(wands_from_service):
            w = wands_from_service[wand_idx - 1]
            if isinstance(w, dict):
                lines.append(f"- {w.get('name','未知')}: max_mp={w.get('max_mp')}, mp_recover={w.get('mp_recover')}, shoot_interval={w.get('shoot_interval')}, cooldown={w.get('cooldown')}")
        detected = (wand_cache.get("detected_wands") or [])
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

    def get_wand_slot_counts(wand_idx: int) -> (Optional[int], Optional[int]):
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

    def verify_state_before_action(expected_wand: int) -> bool:
        # 切换到预期法杖后查询接口，简单等待
        press_wand_slot(expected_wand)
        payload = fetch_spells_payload()
        return payload is not None

    def drag_construct_for_wand(wand_idx: int, selection: List[int]):
        print(f"\n[构筑-法杖{wand_idx}] 准备拖拽 {len(selection)} 个法术...")
        if not sa.used_slot_coordinates or len(sa.used_slot_coordinates) == 0:
            print("⚠ 旧的已装备槽坐标缺失，尝试使用记录的法术槽位置")

        # 目标坐标来源：优先从 store_wand_cache.json 对应法杖的 slots.normal.positions（client坐标）转换为屏幕坐标
        def _client_to_screen(capturer: GameWindowCapture, x: int, y: int):
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
                for p in normal[:4]:
                    try:
                        cx, cy = int(p["x"]), int(p["y"])
                        sx, sy = _client_to_screen(sa.capturer, cx, cy)
                        targets.append({"x": sx, "y": sy})
                    except Exception:
                        continue
                break
            return targets

        def get_targets_fallback_from_rows(wand_idx_local: int) -> List[Dict[str, int]]:
            # Build from slots_all.json with service counts; wand 1 -> row index 1, wand 2 -> row index 2
            try:
                rows = load_rows_from_json("slots_all.json")
                if not rows or len(rows) < 3:
                    return []
                (w1_n, w1_p), (w2_n, w2_p), _bag_cnt = fetch_spell_counts()
                row_idx = 1 if wand_idx_local == 1 else 2
                row = rows[row_idx]
                if row_idx == 1 and w1_n is not None and w1_p is not None:
                    normal_cnt, post_cnt = w1_n, w1_p
                elif row_idx == 2 and w2_n is not None and w2_p is not None:
                    normal_cnt, post_cnt = w2_n, w2_p
                else:
                    # Fallback: keep existing count by detected points
                    normal_cnt = len(row)
                    post_cnt = 0
                normal, _post = generate_positions_from_counts(
                    row=row,
                    normal_count=normal_cnt,
                    post_count=post_cnt,
                    dx_normal=72,
                    gap_normal_to_post=147,
                )
                targets: List[Dict[str, int]] = []
                for p in normal[:4]:
                    sx, sy = _client_to_screen(sa.capturer, int(p["x"]), int(p["y"]))
                    targets.append({"x": sx, "y": sy})
                return targets
            except Exception:
                return []

        target_coords = get_targets_from_cache(wand_idx)
        if not target_coords:
            # 先用行重建坐标作为回退
            target_coords = get_targets_fallback_from_rows(wand_idx)
            if not target_coords:
                # 再回退到旧的坐标（可能没有考虑间隙，不推荐）
                target_coords = sa.used_slot_coordinates or []
            if not target_coords:
                print("⚠ 无可用的目标坐标，跳过该法杖构筑")
                return
        # 槽位1通常是守护杖灵，跳过第一个坐标，避免覆盖
        usable_targets = target_coords[1:] if len(target_coords) > 1 else []
        # 对选择的法术进行位置排序：被动/法术增强最左，法术飞弹居中，其它主动在右
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
        def sort_selection(sel: List[int]) -> List[int]:
            def sort_key(idx1based: int) -> float:
                if not (1 <= idx1based <= len(bag_spells)):
                    return 1.5
                info = bag_spells[idx1based - 1].get("spell_info", {}) or {}
                if is_spell_enhance_or_passive(info):
                    return 0.0
                if is_magic_missile(info):
                    return 1.0
                return 2.0
            return sorted(sel, key=sort_key)
        # 过滤非法/危险/守护杖灵的选择
        filtered = []
        for idx in selection:
            if not (1 <= idx <= len(bag_spells)):
                continue
            info = bag_spells[idx - 1].get("spell_info", {}) or {}
            nm = info.get("name", "")
            if is_dangerous_spell_info(info, nm) or is_guardian_wand_spirit(nm):
                continue
            filtered.append(idx)
        selection = sort_selection(filtered)
        # 确保至少包含一个法术飞弹类
        has_projectile = any(
            is_projectile_spell((bag_spells[idx - 1].get("spell_info", {}) or {}).get("name", ""))
            or is_magic_missile(bag_spells[idx - 1].get("spell_info", {}) or {})
            for idx in selection
            if 1 <= idx <= len(bag_spells)
        )
        if not has_projectile:
            for i, s in enumerate(bag_spells, start=1):
                info = s.get("spell_info", {}) or {}
                nm = info.get("name", "")
                if (is_magic_missile(info) or is_projectile_spell(nm)) and i not in selection:
                    selection.append(i)
                    break
        # 校验状态
        if not verify_state_before_action(wand_idx):
            print("⚠ 接口校验失败，跳过该法杖构筑")
            return
        # 按顺序把选择的法术拖入装备槽（最多槽位数）
        # 严格限制在 normal 槽数量内，避免拖拽到 post 区域
        n_cnt_local, _ = get_wand_slot_counts(wand_idx)
        if isinstance(n_cnt_local, int) and n_cnt_local >= 0:
            max_slots = min(len(selection), len(usable_targets), max(0, n_cnt_local - 1))
        else:
            max_slots = min(len(selection), len(usable_targets))
        for i in range(max_slots):
            sel_idx_1based = selection[i]
            # 将1-based索引转换到bag_spells
            if not (1 <= sel_idx_1based <= len(bag_spells)):
                continue
            spell_item = bag_spells[sel_idx_1based - 1]
            from_coord = spell_item.get("coordinate", {})
            to_coord = usable_targets[i]
            # 再次校验
            if not verify_state_before_action(wand_idx):
                print("⚠ 接口校验失败（拖拽前），中断构筑")
                return
            # 执行拖拽
            sa.drag_spell(
                from_coord.get('x', 0), from_coord.get('y', 0),
                to_coord.get('x', 0), to_coord.get('y', 0)
            )
            _time.sleep(0.4)
            # 拖拽后再次校验
            verify_state_before_action(wand_idx)

    # 为两个法杖分别决策并构筑
    for wand_idx in (1, 2):
        ws = get_wand_specific_summary(wand_idx)
        equipped_for_prompt = summarize_equipped_spells_for_prompt(wands_from_service)
        n_cnt, p_cnt = get_wand_slot_counts(wand_idx)
        construct_prompt = build_construct_prompt(ws, equipped_for_prompt, bag_summary_for_prompt, target_count=4, normal_count=n_cnt, post_count=p_cnt)
        construct_decision = decide_with_text_model(construct_prompt) or {"selected": [], "reason": "模型无返回"}
        print(f"\n构筑决策（法杖{wand_idx}）:")
        print(json.dumps(construct_decision, ensure_ascii=False, indent=2))
        selection = construct_decision.get("selected", []) or []
        # 去除重复与非法索引，避免拖拽异常
        cleaned = []
        seen_idx = set()
        for idx in selection:
            if not isinstance(idx, int):
                continue
            if idx in seen_idx:
                continue
            if 1 <= idx <= len(bag_spells):
                seen_idx.add(idx)
                cleaned.append(idx)
        selection = cleaned
        drag_construct_for_wand(wand_idx, selection)

    print("\n✓ 整合流程完成（双法杖构筑支持、购买前替换支持）")


if __name__ == "__main__":
    run()


