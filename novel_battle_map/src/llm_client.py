from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional

import requests

LOGGER = logging.getLogger("novel_battle_map.llm_client")


class DeepSeekClient:
    def __init__(self, timeout: int = 30, retries: int = 2) -> None:
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip().rstrip("/")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
        self.timeout = timeout
        self.retries = retries

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Optional[Dict[str, Any]]:
        if not self.available:
            LOGGER.warning("DEEPSEEK_API_KEY missing; LLM extraction disabled.")
            return None

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        for attempt in range(1, self.retries + 2):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code >= 400:
                    LOGGER.warning("DeepSeek API error %s: %s", resp.status_code, resp.text[:300])
                    time.sleep(0.8 * attempt)
                    continue
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                parsed = self._parse_json_resilient(text)
                if parsed is not None:
                    return parsed
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("DeepSeek call failed on attempt %s: %s", attempt, exc)
                time.sleep(0.8 * attempt)
        return None

    def _parse_json_resilient(self, raw: str) -> Optional[Dict[str, Any]]:
        raw = raw.strip()
        if not raw:
            return None

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        cleaned = re.sub(r"^```(?:json)?", "", raw).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
        return None
