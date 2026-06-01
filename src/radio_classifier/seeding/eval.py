"""Recall harness for the seeded fingerprint index.

Reads ``truth.csv`` of the form ``clip.wav,song_id`` (or
``clip.wav,artist,title``), runs the index against each clip, and reports
top-1 recall + per-track scores. SPEC §6.3 sets the gate at ≥ 90 %.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from radio_classifier.fingerprint.audfprint_engine import AudfprintIndex
from radio_classifier.fingerprint.types import FingerprintStatus


@dataclass
class EvalRow:
    clip_path: Path
    truth: str
    matched_track_id: str | None
    matched_artist: str | None
    matched_title: str | None
    status: FingerprintStatus
    correct: bool


@dataclass
class EvalReport:
    rows: list[EvalRow]
    total: int
    correct: int

    @property
    def recall(self) -> float:
        return self.correct / self.total if self.total else 0.0


def load_truth(truth_csv: Path) -> dict[Path, str]:
    """Load truth CSV. Supports either ``clip.wav,song_id`` or
    ``clip.wav,artist,title`` (the latter gets joined as ``artist - title``).
    """
    truth: dict[Path, str] = {}
    base = truth_csv.parent
    with truth_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            clip = Path(row[0])
            if not clip.is_absolute():
                clip = (base / clip).resolve()
            if len(row) == 2:
                truth[clip] = row[1].strip()
            elif len(row) >= 3:
                truth[clip] = f"{row[1].strip()} - {row[2].strip()}"
    return truth


def evaluate(
    index: AudfprintIndex,
    truth: dict[Path, str],
    *,
    case_insensitive: bool = True,
) -> EvalReport:
    rows: list[EvalRow] = []
    correct = 0
    for clip, expected in truth.items():
        result = index.match_file(clip)
        matched_label = _join(result.artist, result.title) or result.track_id
        is_correct = False
        if result.status is FingerprintStatus.match and matched_label is not None:
            if case_insensitive:
                is_correct = _label_match(matched_label, expected)
            else:
                is_correct = matched_label == expected
        rows.append(
            EvalRow(
                clip_path=clip,
                truth=expected,
                matched_track_id=result.track_id,
                matched_artist=result.artist,
                matched_title=result.title,
                status=result.status,
                correct=is_correct,
            )
        )
        if is_correct:
            correct += 1
    return EvalReport(rows=rows, total=len(rows), correct=correct)


def _join(a: str | None, b: str | None) -> str | None:
    if a and b:
        return f"{a} - {b}"
    return a or b


def _label_match(matched: str, expected: str) -> bool:
    return _norm(matched) == _norm(expected)


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("_", " ").split())
