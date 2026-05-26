"""Three-tier funnel orchestrator.

The orchestrator is intentionally light: it accepts already-constructed
classifier objects (Tier 1, Tier 2, Tier 3 transcriber, Tier 3 LLM client,
resolver, optional Shazam) and shepherds one :class:`AudioWindow` through
them, returning a :class:`FunnelResult` that records what happened plus the
:class:`SegmentInput` (if any) for downstream persistence.

Heavy model loading is the caller's responsibility — see ``cli.py`` for how
the live pipeline does it once per process.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Protocol

from radio_classifier.acoustic.types import AcousticLabel, AcousticResult
from radio_classifier.commercials.identity import (
    CommercialIdentityResolver,
    CommercialResolution,
)
from radio_classifier.fingerprint.types import FingerprintResult, FingerprintStatus
from radio_classifier.ingest.windows import AudioWindow
from radio_classifier.music.types import ShazamResult, ShazamStatus
from radio_classifier.persistence.broadcast_store import BroadcastStore
from radio_classifier.segments.normalize import (
    segment_input_for_song,
    segment_input_for_speech,
    segment_input_for_unknown_song,
)
from radio_classifier.segments.types import BroadcastCategory, SegmentInput
from radio_classifier.speech.types import (
    BrandMention,
    SpeechPipelineResult,
    SpeechPipelineStatus,
)


class FunnelStage(str, Enum):
    """Which tier produced the final classification for this window."""

    tier1_fingerprint = "tier1_fingerprint"
    tier2_unknown_song = "tier2_unknown_song"
    shazam_fallback = "shazam_fallback"
    tier3_speech = "tier3_speech"
    skipped = "skipped"


@dataclass
class FunnelResult:
    """Diagnostic + persistable output of one full funnel pass."""

    window_start_utc: str
    stage: FunnelStage
    segment_input: SegmentInput | None
    fingerprint: FingerprintResult | None = None
    acoustic: AcousticResult | None = None
    speech: SpeechPipelineResult | None = None
    shazam: ShazamResult | None = None
    commercial_resolution: CommercialResolution | None = None
    brand_mentions: list[BrandMention] | None = None


class Tier1Classifier(Protocol):
    def match_window(self, window: AudioWindow) -> FingerprintResult: ...


class Tier2Classifier(Protocol):
    def classify(self, window: AudioWindow) -> AcousticResult: ...


class Tier3Pipeline(Protocol):
    def __call__(self, window: AudioWindow) -> SpeechPipelineResult: ...


class ShazamFn(Protocol):
    def __call__(self, window: AudioWindow) -> ShazamResult: ...


@dataclass
class FunnelOrchestrator:
    """Wire the three tiers + optional Shazam + commercial identity resolver.

    ``speech_override_threshold`` rescues windows where YAMNet's argmax label
    is MUSIC but the speech probability is non-trivial — the
    "DJ-talking-over-music-bed" and "station sweep / liner with voiceover"
    cases. Without the override these windows would short-circuit at Tier 2
    as ``tier2_unknown_song`` and never reach Whisper + Ollama. Empirically
    pure-music windows on FM produce ``speech_prob`` in [0.00, 0.05]; live
    voiceovers over a music bed jump to 0.10–0.30. 0.10 is the default floor.
    """

    tier1: Tier1Classifier | None
    tier2: Tier2Classifier | None
    tier3: Tier3Pipeline | None
    resolver: CommercialIdentityResolver | None
    store: BroadcastStore | None = None
    shazam_fn: ShazamFn | None = None
    window_seconds: float = 20.0
    speech_override_threshold: float = 0.10
    suppress_singleton_fingerprint_matches: bool = True
    shazam_recheck_windows: int = 4
    _previous_fingerprint_track_id: str | None = field(default=None, init=False, repr=False)
    _unknown_music_window_count: int = field(default=0, init=False, repr=False)
    _cached_shazam: ShazamResult | None = field(default=None, init=False, repr=False)

    def process(self, window: AudioWindow) -> FunnelResult:
        # Tier 1: audfprint song match.
        fp: FingerprintResult | None = None
        if self.tier1 is not None:
            fp = self.tier1.match_window(window)
            fp = self._suppress_singleton_fingerprint_match(fp)
            if fp.status == FingerprintStatus.match:
                self._reset_unknown_music_run()
                song_id = self._record_song_from_fingerprint(fp)
                seg = segment_input_for_song(
                    window_start_utc=window.window_start_utc,
                    artist=fp.artist,
                    title=fp.title,
                    song_id=song_id,
                    confidence=fp.match_score,
                )
                return FunnelResult(
                    window_start_utc=window.window_start_utc,
                    stage=FunnelStage.tier1_fingerprint,
                    segment_input=seg,
                    fingerprint=fp,
                )

        # Tier 2: acoustic gate.
        ac: AcousticResult | None = None
        if self.tier2 is not None:
            ac = self.tier2.classify(window)

        # "Speech-over-music" override: a MUSIC-labeled window with a
        # meaningful speech_prob is most likely DJ patter or a station sweep
        # over a music bed. Push it down to Tier 3 instead of short-circuiting.
        speech_override = (
            ac is not None
            and ac.label is AcousticLabel.MUSIC
            and ac.speech_prob >= self.speech_override_threshold
        )

        if ac is not None and ac.label is AcousticLabel.MUSIC and not speech_override:
            # Possibly fall back to Shazam.
            sz: ShazamResult | None = None
            if self.shazam_fn is not None:
                sz = self._identify_unknown_music_with_shazam(window)
                if sz.status is ShazamStatus.match:
                    song_id = self._record_song_from_shazam(sz)
                    seg = segment_input_for_song(
                        window_start_utc=window.window_start_utc,
                        artist=sz.artist,
                        title=sz.title,
                        song_id=song_id,
                        confidence=sz.confidence,
                    )
                    return FunnelResult(
                        window_start_utc=window.window_start_utc,
                        stage=FunnelStage.shazam_fallback,
                        segment_input=seg,
                        fingerprint=fp,
                        acoustic=ac,
                        shazam=sz,
                    )
            seg = segment_input_for_unknown_song(
                window_start_utc=window.window_start_utc,
                confidence=ac.music_prob if ac else None,
            )
            return FunnelResult(
                window_start_utc=window.window_start_utc,
                stage=FunnelStage.tier2_unknown_song,
                segment_input=seg,
                fingerprint=fp,
                acoustic=ac,
                shazam=sz,
            )

        # Tier 3: speech transcription + LLM classification.
        self._reset_unknown_music_run()
        # Reached when:
        #   - Tier 2 said SPEECH or OTHER, or
        #   - Tier 2 was disabled (still try transcription on the assumption
        #     the operator wants every window classified).
        if self.tier3 is None:
            return FunnelResult(
                window_start_utc=window.window_start_utc,
                stage=FunnelStage.skipped,
                segment_input=None,
                fingerprint=fp,
                acoustic=ac,
                speech=None,
            )

        sp = self.tier3(window)
        if sp.status is not SpeechPipelineStatus.ok or sp.category is None:
            return FunnelResult(
                window_start_utc=window.window_start_utc,
                stage=FunnelStage.skipped,
                segment_input=None,
                fingerprint=fp,
                acoustic=ac,
                speech=sp,
            )

        commercial_resolution: CommercialResolution | None = None
        commercial_id: int | None = None
        if sp.category is BroadcastCategory.COMMERCIAL and self.resolver is not None:
            commercial_resolution = self.resolver.resolve(
                brand=sp.brand,
                transcript=sp.transcript,
                duration_seconds=self.window_seconds,
                signature=sp.commercial_signature,
            )
            commercial_id = commercial_resolution.commercial_id

        if sp.category is BroadcastCategory.SONG:
            seg = segment_input_for_unknown_song(
                window_start_utc=window.window_start_utc,
                confidence=sp.confidence,
            )
        else:
            seg = segment_input_for_speech(
                window_start_utc=window.window_start_utc,
                category=sp.category,
                brand=sp.brand,
                commercial_id=commercial_id,
                transcript_excerpt=sp.transcript[:512] if sp.transcript else None,
                confidence=sp.confidence,
            )

        return FunnelResult(
            window_start_utc=window.window_start_utc,
            stage=FunnelStage.tier3_speech,
            segment_input=seg,
            fingerprint=fp,
            acoustic=ac,
            speech=sp,
            commercial_resolution=commercial_resolution,
            brand_mentions=list(sp.brand_mentions) if sp.brand_mentions else None,
        )

    # -------------------------------------------------------------- internals
    def _identify_unknown_music_with_shazam(self, window: AudioWindow) -> ShazamResult:
        """Run Shazam sparingly during consecutive unknown-music windows.

        The first unknown-music window calls Shazam. Subsequent windows reuse
        the cached result so a three-minute unknown song does not perform
        dozens of external lookups. Every ``shazam_recheck_windows`` windows,
        the external lookup runs again; if the artist/title changed, the
        returned segment key changes and the reducer closes the previous song.
        """
        should_call = (
            self._cached_shazam is None
            or self._unknown_music_window_count == 0
            or (
                self.shazam_recheck_windows > 0
                and self._unknown_music_window_count % self.shazam_recheck_windows == 0
            )
        )
        self._unknown_music_window_count += 1

        if should_call:
            assert self.shazam_fn is not None
            self._cached_shazam = self.shazam_fn(window)
            return self._cached_shazam

        cached = self._cached_shazam
        if cached is None:
            assert self.shazam_fn is not None
            self._cached_shazam = self.shazam_fn(window)
            return self._cached_shazam

        message = cached.message
        if cached.status is ShazamStatus.match:
            message = "cached shazam match"
        elif message is None:
            message = f"cached shazam {cached.status.value}"
        return replace(cached, window_start_utc=window.window_start_utc, message=message)

    def _reset_unknown_music_run(self) -> None:
        self._unknown_music_window_count = 0
        self._cached_shazam = None

    def _suppress_singleton_fingerprint_match(self, fp: FingerprintResult) -> FingerprintResult:
        """Demote isolated Tier-1 song hits that lack adjacent support.

        audfprint occasionally emits weak one-off matches for an unrelated
        song. Real radio song matches should persist across adjacent overlapping
        windows, so require the current raw match to agree with the previous
        raw match before letting Tier 1 short-circuit the funnel.
        """
        if not self.suppress_singleton_fingerprint_matches:
            return fp

        current_track_id = fp.track_id if fp.status is FingerprintStatus.match else None
        previous_track_id = self._previous_fingerprint_track_id
        self._previous_fingerprint_track_id = current_track_id

        if fp.status is not FingerprintStatus.match:
            return fp
        if current_track_id is not None and current_track_id == previous_track_id:
            return fp

        count = f" count={int(fp.match_score)}" if fp.match_score is not None else ""
        return FingerprintResult(
            status=FingerprintStatus.no_match,
            window_start_utc=fp.window_start_utc,
            match_score=fp.match_score,
            message=(
                "suppressed singleton fingerprint match "
                f"track={fp.track_id!r}{count}; requires adjacent same-track match"
            ),
        )

    def _record_song_from_fingerprint(self, fp: FingerprintResult) -> int | None:
        if self.store is None:
            return None
        return self.store.upsert_song(
            artist=fp.artist,
            title=fp.title,
            audfprint_track_id=fp.track_id,
            source="audfprint",
        )

    def _record_song_from_shazam(self, sz: ShazamResult) -> int | None:
        if self.store is None:
            return None
        return self.store.upsert_song(
            artist=sz.artist,
            title=sz.title,
            audfprint_track_id=None,
            source="shazam",
        )
