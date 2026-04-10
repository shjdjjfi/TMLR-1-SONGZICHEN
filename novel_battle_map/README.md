# novel_battle_map

A complete local Python project that reads a war/military narrative `.txt` novel, splits it into chapters, extracts chapter-level battlefield dynamics with an **LLM-first + rules fallback** pipeline, and outputs either:

- a generated procedural abstract battlefield map (`map.png`), or
- `NA` when the chapter does not have enough actionable spatial military dynamics.

The system is designed to be robust, modular, and runnable on Windows/Linux without GIS or real-world geolocation.

---

## 1) Project Overview

### Input
- One `.txt` novel file with military/war narrative chapters.

### Output
For each chapter:
- `chapter_text.txt`
- `result.json` (always produced)
- `map.png` only when `has_battle_map=true`; otherwise `map_image="NA"`

Global:
- `summary.json` with chapter stats and parsing status.

---

## 2) Architecture

```text
main.py
  -> pipeline.BattleMapPipeline
      -> chapter_parser.parse_chapters
      -> battle_detector.HybridBattleExtractor
           -> LLMBattleExtractor (DeepSeek-compatible OpenAI-style API)
           -> RuleBasedBattleExtractor (fallback)
      -> map_generator.ProceduralBattleMapGenerator
      -> output writer (chapter folders + summary.json)
```

### Modules
- `src/models.py`: typed dataclasses for all schema objects.
- `src/chapter_parser.py`: robust English/Chinese chapter splitting + fallback chunking.
- `src/llm_client.py`: DeepSeek-compatible chat completion client using `requests`.
- `src/battle_detector.py`: extractor interfaces + LLM extractor + rule fallback + hybrid orchestration.
- `src/map_generator.py`: procedural operational-style abstract battlefield map rendering.
- `src/pipeline.py`: end-to-end processing and output persistence.
- `src/utils.py`: logging, seed, JSON/text save, `.env` loading helper.

---

## 3) Environment Setup

- Python 3.11+
- Works locally offline except optional LLM API call.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

## 4) `.env` Configuration

Copy and edit:

```bash
cp .env.example .env
```

`.env.example`:

```ini
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

If `.env` or key is missing, the system logs a warning and automatically uses fallback rules.

---

## 5) DeepSeek API Configuration

This project uses OpenAI-style `POST /chat/completions` via `requests`, configured by env vars:

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_MODEL`

Behavior:
- low temperature extraction
- timeout + retries
- resilient JSON parsing
- malformed JSON repair attempt
- if still invalid: fallback to rule extractor

No secrets are hardcoded.

---

## 6) Fallback Behavior

Fallback extractor (`RuleBasedBattleExtractor`) uses weighted signals:

- military entities
- action verbs (advance/retreat/attack/defend/etc.)
- terrain cues (river/bridge/hill/forest/road/village/etc.)
- temporal progression cues

It decides map-worthiness conservatively and always returns valid structured output.

`HybridBattleExtractor` mode labels in `result.json`:
- `"llm"`
- `"fallback_rules"`

---

## 7) Installation

```bash
pip install -r requirements.txt
```

---

## 8) Run Command

```bash
python main.py --input input/sample_novel.txt --output output/
```

Optional flags:
- `--max-chapters 3`
- `--seed 42`
- `--verbose`
- `--disable-llm` (force rule extraction)

Example forcing local fallback only:

```bash
python main.py --input input/sample_novel.txt --output output/ --disable-llm --verbose
```

---

## 9) Output Explanation

Expected output tree:

```text
output/
  chapter_001/
    chapter_text.txt
    result.json
    map.png
  chapter_002/
    chapter_text.txt
    result.json
    map.png
  chapter_003/
    chapter_text.txt
    result.json
  summary.json
```

### `result.json` guarantees
- always exists per chapter
- always has required schema fields
- `map_image` is either `"map.png"` or `"NA"`

### `summary.json` contains
- input path
- chapter count
- map count / NA count
- parsing fallback status (`used_fallback_segmentation`)
- per chapter: id/title/has_map/extraction_mode/output dir

---

## 10) Procedural Map Design

Map generator is fully synthetic (non-GIS):

- 2D elevation field from Gaussian bumps
- contour lines + subtle terrain shading
- curved river
- forest clusters
- roads
- bridge at river crossing
- village marker
- side-colored unit markers (red/blue/neutral)
- tactical overlays:
  - directional arrows for advance/attack
  - dashed lines for retreat/withdraw
  - engagement zones
  - encirclement arcs for surround events

Semantic placement heuristics tie events/terrain hints to visual locations.

---

## 11) Sample Data Behavior

`input/sample_novel.txt` includes 3 chapters:
- Chapter 1: clear river-crossing battle -> map
- Chapter 2 (Chinese): clear tactical operations -> map
- Chapter 3: reflective/dialogue chapter -> NA

This supports a stable demo of **2 maps + 1 NA**.

---

## 12) Future Extensions

- multi-chapter tactical animation/video export
- stronger spatial reasoning and event chaining
- stronger LLM extraction with tool-augmented validators
- web UI for chapter browsing and map interaction
- chapter timeline playback with stepwise tactical states

