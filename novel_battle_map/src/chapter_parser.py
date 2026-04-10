from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from src.models import Chapter


@dataclass
class ChapterParseResult:
    chapters: List[Chapter]
    used_fallback_segmentation: bool


CHAPTER_PATTERNS = [
    re.compile(r"^\s*(chapter\s+\d+[^\n]*)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(chapter\s+[ivxlcdm]+[^\n]*)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(第\s*[0-9一二三四五六七八九十百千]+\s*章[^\n]*)\s*$", re.IGNORECASE),
]


def _detect_headers(lines: List[str]) -> List[Tuple[int, str]]:
    headers: List[Tuple[int, str]] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        for pattern in CHAPTER_PATTERNS:
            if pattern.match(stripped):
                headers.append((idx, stripped))
                break
    return headers


def _fallback_segment(text: str, target_chars: int = 5000) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return [text.strip()] if text.strip() else []

    chunks: List[str] = []
    current: List[str] = []
    total = 0

    for para in paragraphs:
        if total >= target_chars and current:
            chunks.append("\n\n".join(current))
            current = []
            total = 0
        current.append(para)
        total += len(para)

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def parse_chapters(input_path: Path) -> ChapterParseResult:
    text = input_path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
    lines = text.split("\n")
    headers = _detect_headers(lines)

    chapters: List[Chapter] = []
    used_fallback = False

    if len(headers) >= 2:
        for i, (start_idx, title) in enumerate(headers):
            end_idx = headers[i + 1][0] if i + 1 < len(headers) else len(lines)
            section = "\n".join(lines[start_idx + 1 : end_idx]).strip()
            if not section:
                continue
            chapter_id = f"chapter_{len(chapters) + 1:03d}"
            chapters.append(Chapter(chapter_id=chapter_id, chapter_title=title, chapter_text=section))
    elif len(headers) == 1:
        start_idx, title = headers[0]
        section = "\n".join(lines[start_idx + 1 :]).strip()
        if section:
            chapters.append(Chapter(chapter_id="chapter_001", chapter_title=title, chapter_text=section))
    else:
        used_fallback = True
        chunks = _fallback_segment(text)
        for i, chunk in enumerate(chunks, start=1):
            if not chunk.strip():
                continue
            chapters.append(
                Chapter(
                    chapter_id=f"chapter_{i:03d}",
                    chapter_title=f"Segment {i}",
                    chapter_text=chunk.strip(),
                )
            )

    if not chapters and text.strip():
        used_fallback = True
        chapters = [
            Chapter(
                chapter_id="chapter_001",
                chapter_title="Segment 1",
                chapter_text=text.strip(),
            )
        ]

    return ChapterParseResult(chapters=chapters, used_fallback_segmentation=used_fallback)
