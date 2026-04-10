from __future__ import annotations

import abc
import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.llm_client import DeepSeekClient
from src.models import BattleEvent, BattleExtractionResult, MilitaryEntity, TerrainElements

LOGGER = logging.getLogger("novel_battle_map.battle_detector")


class BaseBattleExtractor(abc.ABC):
    @abc.abstractmethod
    def extract(self, chapter_text: str, chapter_title: str) -> BattleExtractionResult:
        raise NotImplementedError


class RuleBasedBattleExtractor(BaseBattleExtractor):
    UNIT_KEYWORDS = [
        "army", "regiment", "battalion", "brigade", "cavalry", "artillery", "troops", "defenders", "scouts",
        "vanguard", "rear guard", "fleet", "师", "团", "营", "连", "骑兵", "炮兵", "守军", "敌军", "援军", "部队", "先锋", "旅",
    ]
    ACTION_KEYWORDS = [
        "advance", "retreat", "attack", "defend", "surround", "cross", "bombard", "converge", "reinforce", "withdraw",
        "engage", "ambush", "occupy", "advanced", "attacked", "withdrew", "crossing", "converged", "occupied",
        "推进", "后撤", "进攻", "防守", "包围", "渡河", "炮击", "集结", "占领", "逼近", "交火", "突袭", "阻击",
    ]
    TERRAIN_KEYWORDS = [
        "hill", "ridge", "river", "ford", "bridge", "forest", "village", "road", "valley", "fort", "trench", "position",
        "高地", "山脊", "河流", "河岸", "渡口", "桥", "森林", "村庄", "道路", "山口", "堡垒", "阵地", "壕沟",
    ]
    TEMPORAL_KEYWORDS = [
        "then", "later", "at dawn", "at dusk", "after several hours", "soon after", "by nightfall", "随后", "接着", "不久", "黎明", "黄昏", "夜间", "数小时后",
    ]

    EVENT_MAP = {
        "advance": ["advance", "advanced", "逼近", "推进", "converge", "converged"],
        "retreat": ["retreat", "withdraw", "withdrew", "后撤"],
        "attack": ["attack", "attacked", "engage", "ambush", "炮击", "进攻", "交火", "突袭"],
        "defend": ["defend", "defenders", "防守", "阻击"],
        "surround": ["surround", "包围"],
        "cross": ["cross", "crossing", "ford", "渡河"],
        "occupy": ["occupy", "occupied", "占领"],
    }

    def extract(self, chapter_text: str, chapter_title: str) -> BattleExtractionResult:
        text_lower = chapter_text.lower()
        unit_hits = self._keyword_hits(chapter_text, self.UNIT_KEYWORDS, lower=text_lower)
        action_hits = self._keyword_hits(chapter_text, self.ACTION_KEYWORDS, lower=text_lower)
        terrain_hits = self._keyword_hits(chapter_text, self.TERRAIN_KEYWORDS, lower=text_lower)
        temporal_hits = self._keyword_hits(chapter_text, self.TEMPORAL_KEYWORDS, lower=text_lower)

        score = 2.2 * len(action_hits) + 1.7 * len(terrain_hits) + 1.2 * len(unit_hits) + 0.9 * len(temporal_hits)
        has_battle_map = score >= 7.0 and len(action_hits) >= 2 and (len(terrain_hits) >= 1 or len(unit_hits) >= 2)

        terrain = TerrainElements(
            has_river=self._contains_any(chapter_text, ["river", "河流", "河岸", "渡口", "ford"]),
            has_forest=self._contains_any(chapter_text, ["forest", "森林"]),
            has_road=self._contains_any(chapter_text, ["road", "道路", "route"]),
            has_bridge=self._contains_any(chapter_text, ["bridge", "桥"]),
            has_hill=self._contains_any(chapter_text, ["hill", "ridge", "高地", "山脊"]),
            has_village=self._contains_any(chapter_text, ["village", "村庄", "town", "strongpoint", "堡垒"]),
        )

        entities = self._extract_entities(chapter_text)
        events = self._extract_events(chapter_text, entities)

        if not has_battle_map:
            reason = "The chapter lacks sufficient actionable battlefield movement and spatial military dynamics."
            entities = entities[:2] if entities else []
            events = []
        else:
            reason = "The chapter includes military movement/engagement with terrain-aware action suitable for tactical mapping."
            if not events and entities:
                events = [
                    BattleEvent(
                        order=1,
                        event_type="advance",
                        actor=entities[0].name,
                        location_hint="frontline approach",
                        terrain_hint="mixed terrain",
                        intensity="medium",
                    )
                ]

        return BattleExtractionResult(
            chapter_title=chapter_title,
            has_battle_map=has_battle_map,
            reason=reason,
            extraction_mode="fallback_rules",
            terrain_elements=terrain,
            military_entities=entities,
            events=events,
        )

    def _keyword_hits(self, text: str, keywords: Sequence[str], lower: Optional[str] = None) -> List[str]:
        hay = lower if lower is not None else text.lower()
        hits: List[str] = []
        for kw in keywords:
            if any(ord(ch) > 127 for ch in kw):
                if kw in text:
                    hits.append(kw)
            else:
                pattern = rf"\b{re.escape(kw.lower())}\w*\b" if re.match(r"^[a-z]+$", kw.lower()) else rf"\b{re.escape(kw.lower())}\b"
                if re.search(pattern, hay):
                    hits.append(kw)
        return hits

    def _contains_any(self, text: str, keywords: Sequence[str]) -> bool:
        hits = self._keyword_hits(text, keywords, lower=text.lower())
        return bool(hits)

    def _extract_entities(self, text: str) -> List[MilitaryEntity]:
        patterns = [
            r"\b([A-Z][a-z]+\s+(?:regiment|brigade|battalion|army|division|fleet))\b",
            r"\b((?:northern|southern|eastern|western)\s+(?:regiment|brigade|army|troops))\b",
            r"([\u4e00-\u9fff]{2,8}(?:师|团|营|连|旅|守军|敌军|援军|部队|先锋))",
        ]
        found: Counter[str] = Counter()
        for pattern in patterns:
            for m in re.findall(pattern, text):
                found[m.strip()] += 1

        entities: List[MilitaryEntity] = []
        for name, _ in found.most_common(8):
            side = "unknown"
            lowered = name.lower()
            if any(s in lowered for s in ["northern", "red", "north"]):
                side = "red"
            elif any(s in lowered for s in ["southern", "blue", "south"]):
                side = "blue"
            elif any(s in name for s in ["敌", "蓝"]):
                side = "blue"
            elif any(s in name for s in ["我", "红", "北"]):
                side = "red"
            entities.append(MilitaryEntity(name=name, side=side, type="unit"))

        if not entities:
            # Minimal recovery from generic mentions
            if self._contains_any(text, ["troops", "部队", "守军", "敌军"]):
                entities.append(MilitaryEntity(name="main force", side="unknown", type="unit"))
        return entities

    def _extract_events(self, text: str, entities: List[MilitaryEntity]) -> List[BattleEvent]:
        lines = [ln.strip() for ln in re.split(r"[。.!?\n]", text) if ln.strip()]
        events: List[BattleEvent] = []
        entity_names = [e.name for e in entities]

        for line in lines:
            event_type = self._classify_event(line)
            if not event_type:
                continue
            actor = self._find_actor(line, entity_names)
            terrain_hint = self._line_terrain_hint(line)
            location_hint = self._line_location_hint(line)
            intensity = "high" if re.search(r"heavy|fierce|激烈|猛|持续炮击", line, re.IGNORECASE) else "medium"
            events.append(
                BattleEvent(
                    order=len(events) + 1,
                    event_type=event_type,
                    actor=actor,
                    target=None,
                    location_hint=location_hint,
                    terrain_hint=terrain_hint,
                    intensity=intensity,
                )
            )
            if len(events) >= 8:
                break
        return events

    def _classify_event(self, line: str) -> Optional[str]:
        lline = line.lower()
        for etype, kws in self.EVENT_MAP.items():
            for kw in kws:
                if any(ord(ch) > 127 for ch in kw):
                    if kw in line:
                        return etype
                elif re.search(rf"\b{re.escape(kw)}\b", lline):
                    return etype
        return None

    def _find_actor(self, line: str, entity_names: List[str]) -> str:
        for name in entity_names:
            if name in line:
                return name
        return entity_names[0] if entity_names else "unknown force"

    def _line_terrain_hint(self, line: str) -> Optional[str]:
        mapping = {
            "river": ["river", "河", "渡口", "ford"],
            "bridge": ["bridge", "桥"],
            "road": ["road", "道路"],
            "forest": ["forest", "森林"],
            "hill": ["hill", "ridge", "高地", "山脊"],
            "village": ["village", "村庄", "town", "堡垒"],
        }
        for label, kws in mapping.items():
            if any((kw in line if any(ord(ch) > 127 for ch in kw) else re.search(rf"\b{re.escape(kw)}\b", line.lower())) for kw in kws):
                return label
        return None

    def _line_location_hint(self, line: str) -> Optional[str]:
        m = re.search(r"(?:toward|near|at|to|around)\s+([a-z\- ]{3,40})", line, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m_cn = re.search(r"在(.{2,12}?)(?:附近|一线|周边|处)", line)
        if m_cn:
            return m_cn.group(1)
        return None


class LLMBattleExtractor(BaseBattleExtractor):
    def __init__(self, client: Optional[DeepSeekClient] = None) -> None:
        self.client = client or DeepSeekClient()

    def extract(self, chapter_text: str, chapter_title: str) -> BattleExtractionResult:
        system_prompt = (
            "You are a military narrative information extractor. Output JSON only. "
            "No markdown and no extra prose."
        )
        user_prompt = self._build_prompt(chapter_text, chapter_title)
        parsed = self.client.chat_json(system_prompt, user_prompt, temperature=0.1)
        if parsed is None:
            raise ValueError("LLM extraction failed")

        valid = self._validate_and_repair(parsed, chapter_title)
        if valid is None:
            repair_prompt = (
                "Repair the following into valid strict JSON matching required schema. Return JSON only:\n"
                + json.dumps(parsed, ensure_ascii=False)
            )
            parsed2 = self.client.chat_json(system_prompt, repair_prompt, temperature=0.0)
            if parsed2 is None:
                raise ValueError("LLM JSON repair failed")
            valid = self._validate_and_repair(parsed2, chapter_title)
            if valid is None:
                raise ValueError("LLM returned invalid schema after repair")
        valid.extraction_mode = "llm"
        return valid

    def _build_prompt(self, chapter_text: str, chapter_title: str) -> str:
        schema_note = {
            "chapter_title": "string",
            "has_battle_map": "bool",
            "reason": "string",
            "terrain_elements": {
                "has_river": "bool",
                "has_forest": "bool",
                "has_road": "bool",
                "has_bridge": "bool",
                "has_hill": "bool",
                "has_village": "bool",
            },
            "military_entities": [{"name": "string", "side": "red|blue|unknown", "type": "unit|commander|fortification"}],
            "events": [{
                "order": "int",
                "event_type": "advance|retreat|attack|defend|surround|cross|bombard|engage|occupy|reinforce|withdraw",
                "actor": "string",
                "target": "string|null",
                "location_hint": "string|null",
                "terrain_hint": "string|null",
                "intensity": "low|medium|high",
            }],
        }
        return (
            "Task: Extract chapter-local battlefield dynamics conservatively.\n"
            "If chapter is mostly dialogue/exposition/internal thoughts without actionable spatial military dynamics, set has_battle_map=false.\n"
            "If chapter includes movement/positioning/attack-defense/retreat/crossing/flanking/engagement/occupation with terrain relevance, set has_battle_map=true.\n"
            "Avoid hallucinations. Infer lightly only when strongly implied.\n"
            f"Chapter title: {chapter_title}\n"
            f"Required JSON schema shape: {json.dumps(schema_note, ensure_ascii=False)}\n"
            "Return valid JSON only.\n"
            f"Chapter text:\n{chapter_text[:12000]}"
        )

    def _validate_and_repair(self, data: Dict[str, Any], chapter_title: str) -> Optional[BattleExtractionResult]:
        required_top = ["chapter_title", "has_battle_map", "reason", "terrain_elements", "military_entities", "events"]
        if any(k not in data for k in required_top):
            return None

        terrain = data.get("terrain_elements")
        if not isinstance(terrain, dict):
            return None

        terrain_obj = TerrainElements(
            has_river=bool(terrain.get("has_river", False)),
            has_forest=bool(terrain.get("has_forest", False)),
            has_road=bool(terrain.get("has_road", False)),
            has_bridge=bool(terrain.get("has_bridge", False)),
            has_hill=bool(terrain.get("has_hill", False)),
            has_village=bool(terrain.get("has_village", False)),
        )

        entities_raw = data.get("military_entities", [])
        events_raw = data.get("events", [])
        if not isinstance(entities_raw, list) or not isinstance(events_raw, list):
            return None

        entities: List[MilitaryEntity] = []
        for item in entities_raw[:20]:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            entities.append(
                MilitaryEntity(
                    name=str(item.get("name")).strip(),
                    side=str(item.get("side", "unknown") or "unknown"),
                    type=str(item.get("type", "unit") or "unit"),
                )
            )

        events: List[BattleEvent] = []
        for idx, item in enumerate(events_raw[:30], start=1):
            if not isinstance(item, dict):
                continue
            actor = str(item.get("actor", "unknown force")).strip() or "unknown force"
            event_type = str(item.get("event_type", "advance")).strip() or "advance"
            try:
                order = int(item.get("order", idx))
            except Exception:  # noqa: BLE001
                order = idx
            events.append(
                BattleEvent(
                    order=order,
                    event_type=event_type,
                    actor=actor,
                    target=item.get("target"),
                    location_hint=item.get("location_hint"),
                    terrain_hint=item.get("terrain_hint"),
                    intensity=str(item.get("intensity", "medium") or "medium"),
                )
            )

        try:
            has_battle_map = bool(data["has_battle_map"])
        except Exception:  # noqa: BLE001
            return None

        ch_title = str(data.get("chapter_title") or chapter_title)
        reason = str(data.get("reason") or "No reason provided.")

        if has_battle_map and not events and not entities:
            return None

        return BattleExtractionResult(
            chapter_title=ch_title,
            has_battle_map=has_battle_map,
            reason=reason,
            extraction_mode="llm",
            terrain_elements=terrain_obj,
            military_entities=entities,
            events=events,
        )


class HybridBattleExtractor(BaseBattleExtractor):
    def __init__(self, disable_llm: bool = False) -> None:
        self.disable_llm = disable_llm
        self.llm = LLMBattleExtractor()
        self.rules = RuleBasedBattleExtractor()

    def extract(self, chapter_text: str, chapter_title: str) -> BattleExtractionResult:
        if self.disable_llm:
            return self.rules.extract(chapter_text, chapter_title)

        try:
            result = self.llm.extract(chapter_text, chapter_title)
            result.extraction_mode = "llm"
            return result
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("LLM extraction failed for '%s': %s. Falling back to rules.", chapter_title, exc)
            result = self.rules.extract(chapter_text, chapter_title)
            result.extraction_mode = "fallback_rules"
            return result
