"""Ollama HTTP client for transcript → 5-class JSON classification.

Behavior harvested from ``live105sux/src/live105sux/speech/ollama.py`` and
adapted to the new schema in :mod:`...json_schema`.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from json import JSONDecodeError

from pydantic import ValidationError

from radio_classifier.speech.json_schema import LlmClassificationJson
from radio_classifier.speech.prompts import FEW_SHOTS, SYSTEM_PROMPT


class OllamaClassificationError(Exception):
    """Failed to obtain valid 5-class JSON from Ollama after retries."""


def _default_base_url() -> str:
    return os.environ.get(
        "RADIO_CLASSIFIER_OLLAMA_HOST", "http://127.0.0.1:11434"
    ).rstrip("/")


def _default_model() -> str:
    return os.environ.get(
        "RADIO_CLASSIFIER_OLLAMA_MODEL", "llama3.2:latest"
    )


def _default_keep_alive() -> int | str:
    """How long Ollama should keep the model resident between requests.

    Ollama unloads idle models after 5 minutes by default, so long capture
    runs reload the model (hundreds of MB) at every block boundary — the
    reload churn that drives the WSL2 VM into swap and pegs the host's HDD.

    Pinning the model resident eliminates that. The value is whatever Ollama's
    ``keep_alive`` accepts: ``-1`` (or any negative int) keeps it loaded
    indefinitely, ``0`` unloads immediately, a positive int is seconds, and a
    duration string like ``"30m"`` also works. Default: ``-1`` (stay loaded).
    """
    raw = os.environ.get("RADIO_CLASSIFIER_OLLAMA_KEEP_ALIVE", "-1").strip()
    try:
        return int(raw)
    except ValueError:
        return raw


def _build_messages(text: str, include_fewshots: bool = True) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if include_fewshots:
        for ex in FEW_SHOTS:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
    messages.append({"role": "user", "content": text})
    return messages


class OllamaSpeechClassifier:
    """POST ``/api/chat`` to a local Ollama server, expect strict JSON back."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        include_fewshots: bool = True,
        request_timeout: float = 120.0,
        keep_alive: int | str | None = None,
    ) -> None:
        raw = base_url if base_url is not None else _default_base_url()
        self.base_url = raw.rstrip("/")
        self.model = model if model is not None else _default_model()
        self.include_fewshots = include_fewshots
        self.request_timeout = request_timeout
        self.keep_alive = keep_alive if keep_alive is not None else _default_keep_alive()
        self._cache: dict[str, LlmClassificationJson] = {}

    def classify_transcript(self, text: str) -> LlmClassificationJson:
        """Return validated 5-class JSON; retry up to 3 times on failure."""
        cache_key = _cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                result = self._classify_once(text)
                self._cache[cache_key] = result
                return result
            except (
                OSError,
                urllib.error.URLError,
                urllib.error.HTTPError,
                JSONDecodeError,
                ValidationError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                last_err = exc
                if attempt < 2:
                    time.sleep(0.25 if attempt == 0 else 0.5)
        msg = f"ollama classification failed after retries: {last_err!s}"
        raise OllamaClassificationError(msg) from last_err

    def _classify_once(self, text: str) -> LlmClassificationJson:
        url = f"{self.base_url}/api/chat"
        body = {
            "model": self.model,
            "messages": _build_messages(text, include_fewshots=self.include_fewshots),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
            "keep_alive": self.keep_alive,
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
            raw = resp.read().decode("utf-8")
        payload = json.loads(raw)
        content = payload["message"]["content"]
        if isinstance(content, dict):
            obj = content
        else:
            obj = json.loads(content)
        return LlmClassificationJson.model_validate(obj)


def _cache_key(text: str) -> str:
    return " ".join(text.lower().split())
