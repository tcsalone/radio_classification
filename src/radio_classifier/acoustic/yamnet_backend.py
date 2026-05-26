"""YAMNet (TensorFlow Hub) — 3-way roll-up over AudioSet logits.

The YAMNet model returns per-frame scores for 521 AudioSet classes. We:

1. Load the model + class map ONCE per process (slow; ~2s on GPU).
2. Resample each window to 16 kHz mono float32 in [-1, 1].
3. Get per-frame scores, average across frames.
4. Sum probabilities into three buckets — ``MUSIC``, ``SPEECH``, ``OTHER`` —
   using the class-name groupings in :data:`_MUSIC_PREFIXES` and
   :data:`_SPEECH_PREFIXES`.
5. Apply routing rules in :func:`route_label`.

The fine class names are *only* used for diagnostics in
``AcousticResult.top_classes``; they are not exposed to downstream tiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from radio_classifier.acoustic.resample import (
    pcm_int16_to_float32_normalized,
    resample_pcm_int16_to_16k,
)
from radio_classifier.acoustic.types import AcousticLabel, AcousticResult
from radio_classifier.ingest.windows import AudioWindow


_YAMNET_HUB_URL = "https://tfhub.dev/google/yamnet/1"


# AudioSet class-name prefixes that count as MUSIC. The full hierarchy starts
# with "Music" as the root node; we accept any descendant of it plus a few
# common instrument categories that AudioSet places outside that subtree.
_MUSIC_PREFIXES: tuple[str, ...] = (
    "Music",
    "Singing",
    "Yodeling",
    "Choir",
    "Rapping",
    "Beatboxing",
    "Drum",
    "Drum kit",
    "Drum machine",
    "Snare drum",
    "Bass drum",
    "Cymbal",
    "Hi-hat",
    "Guitar",
    "Bass guitar",
    "Electric guitar",
    "Acoustic guitar",
    "Banjo",
    "Mandolin",
    "Ukulele",
    "Violin, fiddle",
    "Cello",
    "Double bass",
    "Harp",
    "Piano",
    "Keyboard (musical)",
    "Organ",
    "Synthesizer",
    "Trumpet",
    "Trombone",
    "Saxophone",
    "Flute",
    "Clarinet",
    "Harmonica",
    "Accordion",
    "Bagpipes",
    "Steel guitar, slide guitar",
    "Ambient music",
    "Background music",
    "Pop music",
    "Rock music",
    "Heavy metal",
    "Hip hop music",
    "Country",
    "Soul music",
    "Blues",
    "Jazz",
    "Funk",
    "Reggae",
    "Folk music",
    "Electronic music",
    "Dance music",
    "Punk rock",
    "House music",
    "Techno",
    "Dubstep",
    "Trance music",
)

_SPEECH_PREFIXES: tuple[str, ...] = (
    "Speech",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Conversation",
    "Narration, monologue",
    "Babbling",
    "Whispering",
    "Shout",
    "Yell",
)


def _starts_with_any(name: str, prefixes: tuple[str, ...]) -> bool:
    nm = name.strip()
    return any(nm == p or nm.startswith(p) for p in prefixes)


def route_label(
    music_prob: float,
    speech_prob: float,
    other_prob: float,
    *,
    min_prob: float = 0.25,
    speech_bias: bool = True,
) -> AcousticLabel:
    """Choose a 3-way label given the bucketed probabilities.

    Rules (in order):

    1. If max bucket below ``min_prob`` → ``SPEECH`` (Tier 3 will handle ambiguity).
    2. If ``speech_bias`` AND ``music_prob`` is dominant but ``speech_prob`` is
       still substantial (≥ 0.35 of summed mass, i.e. comparable order of
       magnitude) → ``SPEECH`` (DJ-talk-over-music wins for transcription).
    3. Otherwise argmax of the three buckets.
    """
    probs = {
        AcousticLabel.MUSIC: music_prob,
        AcousticLabel.SPEECH: speech_prob,
        AcousticLabel.OTHER: other_prob,
    }
    if max(probs.values()) < min_prob:
        return AcousticLabel.SPEECH
    total = music_prob + speech_prob + other_prob
    if (
        speech_bias
        and total > 0
        and (music_prob / total) >= 0.35
        and (speech_prob / total) >= 0.35
    ):
        return AcousticLabel.SPEECH
    return max(probs.items(), key=lambda kv: kv[1])[0]


@dataclass
class _YamnetModel:
    """Loaded YAMNet model + class-name table + bucket index sets."""

    model: Any
    class_names: list[str]
    music_idx: np.ndarray
    speech_idx: np.ndarray
    other_idx: np.ndarray


def _load_yamnet() -> _YamnetModel:
    # Best-effort: ensure pip-installed CUDA libs are visible to TensorFlow.
    # When the [gpu] extra is installed this makes YAMNet run on GPU; when it
    # isn't, the preload is a no-op and TF transparently falls back to CPU.
    from radio_classifier.gpu import preload_nvidia_libs

    preload_nvidia_libs()

    import tensorflow_hub as hub  # type: ignore

    model = hub.load(_YAMNET_HUB_URL)
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    class_names = _read_class_map(class_map_path)
    music_idx = np.array(
        [i for i, n in enumerate(class_names) if _starts_with_any(n, _MUSIC_PREFIXES)],
        dtype=np.int64,
    )
    speech_idx = np.array(
        [i for i, n in enumerate(class_names) if _starts_with_any(n, _SPEECH_PREFIXES)],
        dtype=np.int64,
    )
    all_idx = np.arange(len(class_names), dtype=np.int64)
    accounted = np.concatenate([music_idx, speech_idx])
    other_idx = np.setdiff1d(all_idx, accounted, assume_unique=False)
    return _YamnetModel(
        model=model,
        class_names=class_names,
        music_idx=music_idx,
        speech_idx=speech_idx,
        other_idx=other_idx,
    )


def _read_class_map(class_map_path: str) -> list[str]:
    """Parse YAMNet's class_map CSV: ``index,mid,display_name``."""
    import csv

    names: list[str] = []
    with open(class_map_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return []
        try:
            name_col = header.index("display_name")
        except ValueError:
            name_col = -1
        for row in reader:
            if not row:
                continue
            names.append(row[name_col].strip())
    return names


class YamnetAcousticClassifier:
    """Stateful YAMNet wrapper — loads once, classifies many windows.

    ``min_prob`` and ``speech_bias`` are exposed because Phase E validation
    is expected to tune them on a real FM corpus.
    """

    def __init__(
        self,
        *,
        min_prob: float = 0.25,
        speech_bias: bool = True,
    ) -> None:
        self._yam = _load_yamnet()
        self.min_prob = min_prob
        self.speech_bias = speech_bias

    def classify(self, window: AudioWindow) -> AcousticResult:
        pcm_16k = resample_pcm_int16_to_16k(window.samples, window.sample_rate_hz)
        waveform = pcm_int16_to_float32_normalized(pcm_16k)

        scores_tensor, _embeddings, _spectrogram = self._yam.model(waveform)
        scores = scores_tensor.numpy()  # shape: (num_frames, 521)
        mean_scores = scores.mean(axis=0)  # (521,)
        mean_scores = np.clip(mean_scores, 0.0, 1.0)

        music_mass = float(mean_scores[self._yam.music_idx].sum()) if self._yam.music_idx.size else 0.0
        speech_mass = float(mean_scores[self._yam.speech_idx].sum()) if self._yam.speech_idx.size else 0.0
        other_mass = float(mean_scores[self._yam.other_idx].sum()) if self._yam.other_idx.size else 0.0
        # Normalize to a 3-way distribution.
        total = music_mass + speech_mass + other_mass
        if total <= 0:
            music_p = speech_p = other_p = 1.0 / 3.0
        else:
            music_p = music_mass / total
            speech_p = speech_mass / total
            other_p = other_mass / total

        # Top-5 fine classes for diagnostics.
        top_idx = np.argsort(-mean_scores)[:5]
        top_classes = [
            (self._yam.class_names[int(i)], float(mean_scores[int(i)])) for i in top_idx
        ]

        label = route_label(
            music_p,
            speech_p,
            other_p,
            min_prob=self.min_prob,
            speech_bias=self.speech_bias,
        )

        return AcousticResult(
            label=label,
            window_start_utc=window.window_start_utc,
            music_prob=music_p,
            speech_prob=speech_p,
            other_prob=other_p,
            top_classes=top_classes,
        )
