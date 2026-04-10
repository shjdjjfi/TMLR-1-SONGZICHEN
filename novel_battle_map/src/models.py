from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Chapter:
    chapter_id: str
    chapter_title: str
    chapter_text: str


@dataclass
class TerrainElements:
    has_river: bool = False
    has_forest: bool = False
    has_road: bool = False
    has_bridge: bool = False
    has_hill: bool = False
    has_village: bool = False


@dataclass
class MilitaryEntity:
    name: str
    side: str = "unknown"
    type: str = "unit"


@dataclass
class BattleEvent:
    order: int
    event_type: str
    actor: str
    target: Optional[str] = None
    location_hint: Optional[str] = None
    terrain_hint: Optional[str] = None
    intensity: str = "medium"


@dataclass
class BattleExtractionResult:
    chapter_title: str
    has_battle_map: bool
    reason: str
    extraction_mode: str
    terrain_elements: TerrainElements = field(default_factory=TerrainElements)
    military_entities: List[MilitaryEntity] = field(default_factory=list)
    events: List[BattleEvent] = field(default_factory=list)

    def to_result_json(self, chapter_id: str, map_image: str) -> Dict[str, Any]:
        return {
            "chapter_id": chapter_id,
            "chapter_title": self.chapter_title,
            "has_battle_map": self.has_battle_map,
            "reason": self.reason,
            "extraction_mode": self.extraction_mode,
            "terrain_elements": {
                "has_river": self.terrain_elements.has_river,
                "has_forest": self.terrain_elements.has_forest,
                "has_road": self.terrain_elements.has_road,
                "has_bridge": self.terrain_elements.has_bridge,
                "has_hill": self.terrain_elements.has_hill,
                "has_village": self.terrain_elements.has_village,
            },
            "military_entities": [entity.__dict__ for entity in self.military_entities],
            "events": [event.__dict__ for event in self.events],
            "map_image": map_image,
        }


@dataclass
class ChapterProcessingResult:
    chapter_id: str
    chapter_title: str
    has_battle_map: bool
    extraction_mode: str
    output_dir: Path
