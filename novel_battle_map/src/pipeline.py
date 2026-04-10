from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.battle_detector import HybridBattleExtractor
from src.chapter_parser import parse_chapters
from src.map_generator import ProceduralBattleMapGenerator
from src.models import ChapterProcessingResult
from src.utils import ensure_dir, save_json, save_text

LOGGER = logging.getLogger("novel_battle_map.pipeline")


@dataclass
class PipelineConfig:
    input_file: Path
    output_dir: Path
    max_chapters: Optional[int] = None
    seed: int = 42
    disable_llm: bool = False


class BattleMapPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.extractor = HybridBattleExtractor(disable_llm=config.disable_llm)
        self.map_generator = ProceduralBattleMapGenerator(seed=config.seed)

    def run(self) -> dict:
        ensure_dir(self.config.output_dir)
        parse_result = parse_chapters(self.config.input_file)
        chapters = parse_result.chapters
        if self.config.max_chapters is not None:
            chapters = chapters[: self.config.max_chapters]

        chapter_summaries: List[ChapterProcessingResult] = []
        map_count = 0

        for chapter in chapters:
            chapter_out = ensure_dir(self.config.output_dir / chapter.chapter_id)
            save_text(chapter.chapter_text, chapter_out / "chapter_text.txt")

            extraction = self.extractor.extract(chapter.chapter_text, chapter.chapter_title)
            map_image = "NA"

            if extraction.has_battle_map:
                try:
                    self.map_generator.generate(extraction, chapter_out / "map.png")
                    map_image = "map.png"
                    map_count += 1
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Map generation failed for %s: %s", chapter.chapter_id, exc)
                    extraction.has_battle_map = False
                    extraction.reason = f"Map generation failed safely; exported NA. Root cause: {type(exc).__name__}."
                    map_image = "NA"

            result_json = extraction.to_result_json(chapter.chapter_id, map_image)
            save_json(result_json, chapter_out / "result.json")

            chapter_summaries.append(
                ChapterProcessingResult(
                    chapter_id=chapter.chapter_id,
                    chapter_title=chapter.chapter_title,
                    has_battle_map=extraction.has_battle_map and map_image == "map.png",
                    extraction_mode=extraction.extraction_mode,
                    output_dir=chapter_out,
                )
            )

        summary = {
            "input_file": str(self.config.input_file),
            "total_chapters": len(chapters),
            "total_chapters_with_maps": map_count,
            "total_chapters_with_na": len(chapters) - map_count,
            "chapter_parsing": {
                "used_fallback_segmentation": parse_result.used_fallback_segmentation,
            },
            "chapters": [
                {
                    "chapter_id": c.chapter_id,
                    "chapter_title": c.chapter_title,
                    "has_battle_map": c.has_battle_map,
                    "extraction_mode": c.extraction_mode,
                    "output_directory": str(c.output_dir),
                }
                for c in chapter_summaries
            ],
        }
        save_json(summary, self.config.output_dir / "summary.json")
        return summary
