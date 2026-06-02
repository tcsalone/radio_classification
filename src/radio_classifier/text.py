"""Small text similarity helpers used by runtime reducers and cleanup jobs."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> list[str]:
    return [t.casefold() for t in _WORD_RE.findall(text or "")]


def text_similarity(a: str, b: str) -> float:
    """Combined token Jaccard / cosine similarity for ASR transcript text."""

    a_tokens = tokenize(a)
    b_tokens = tokenize(b)
    if not a_tokens or not b_tokens:
        return 0.0
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    jaccard = len(a_set & b_set) / len(a_set | b_set)
    cosine = _cosine(a_tokens, b_tokens)
    return max(jaccard, cosine)


def _cosine(a: Iterable[str], b: Iterable[str]) -> float:
    ca = Counter(a)
    cb = Counter(b)
    shared = set(ca) & set(cb)
    num = sum(ca[t] * cb[t] for t in shared)
    da = math.sqrt(sum(v * v for v in ca.values()))
    db = math.sqrt(sum(v * v for v in cb.values()))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)
