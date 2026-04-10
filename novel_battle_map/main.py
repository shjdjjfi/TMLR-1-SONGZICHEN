from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import BattleMapPipeline, PipelineConfig
from src.utils import load_dotenv_file, setup_logging, set_global_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate procedural battlefield maps from war novel chapters.")
    parser.add_argument("--input", required=True, type=Path, help="Path to input .txt novel file.")
    parser.add_argument("--output", required=True, type=Path, help="Path to output directory.")
    parser.add_argument("--max-chapters", type=int, default=None, help="Optional max number of chapters to process.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible map generation.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs.")
    parser.add_argument("--disable-llm", action="store_true", help="Force rule-based extraction only.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    load_dotenv_file(Path(".env"))
    set_global_seed(args.seed)

    config = PipelineConfig(
        input_file=args.input,
        output_dir=args.output,
        max_chapters=args.max_chapters,
        seed=args.seed,
        disable_llm=args.disable_llm,
    )

    pipeline = BattleMapPipeline(config)
    summary = pipeline.run()
    print(f"Done. Chapters: {summary['total_chapters']}, maps: {summary['total_chapters_with_maps']}, NA: {summary['total_chapters_with_na']}")


if __name__ == "__main__":
    main()
