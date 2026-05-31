"""End-to-end orchestrator test with all tiers mocked."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from radio_classifier.acoustic.types import AcousticLabel, AcousticResult
from radio_classifier.commercials.identity import CommercialIdentityResolver
from radio_classifier.fingerprint.types import FingerprintResult, FingerprintStatus
from radio_classifier.ingest.windows import AudioWindow
from radio_classifier.music.types import ShazamResult, ShazamStatus
from radio_classifier.persistence import BroadcastStore
from radio_classifier.pipeline import FunnelOrchestrator, FunnelStage
from radio_classifier.segments.types import BroadcastCategory
from radio_classifier.speech.types import (
    BrandMention,
    CommercialSignature,
    SpeechPipelineResult,
    SpeechPipelineStatus,
)


def _make_window(start: str = "2020-01-01T00:00:00.000Z") -> AudioWindow:
    return AudioWindow(
        samples=np.zeros(16_000 * 20, dtype=np.int16),
        sample_rate_hz=16_000,
        window_start_utc=start,
        frame_count=16_000 * 20,
    )


@dataclass
class FakeTier1:
    status: FingerprintStatus = FingerprintStatus.no_match
    artist: str | None = None
    title: str | None = None
    track_id: str | None = None

    def match_window(self, window: AudioWindow) -> FingerprintResult:
        return FingerprintResult(
            status=self.status,
            window_start_utc=window.window_start_utc,
            artist=self.artist,
            title=self.title,
            track_id=self.track_id,
            match_score=10.0 if self.status is FingerprintStatus.match else None,
        )


@dataclass
class SequenceTier1:
    results: list[FingerprintResult]

    def match_window(self, window: AudioWindow) -> FingerprintResult:
        if not self.results:
            return FingerprintResult(
                status=FingerprintStatus.no_match,
                window_start_utc=window.window_start_utc,
            )
        result = self.results.pop(0)
        result.window_start_utc = window.window_start_utc
        return result


@dataclass
class FakeTier2:
    label: AcousticLabel = AcousticLabel.SPEECH
    music_prob: float = 0.1
    speech_prob: float = 0.8
    other_prob: float = 0.1

    def classify(self, window: AudioWindow) -> AcousticResult:
        return AcousticResult(
            label=self.label,
            window_start_utc=window.window_start_utc,
            music_prob=self.music_prob,
            speech_prob=self.speech_prob,
            other_prob=self.other_prob,
            top_classes=[],
        )


@dataclass
class SequenceTier2:
    labels: list[AcousticLabel]
    music_prob: float = 0.9
    speech_prob: float = 0.01
    other_prob: float = 0.09

    def classify(self, window: AudioWindow) -> AcousticResult:
        label = self.labels.pop(0) if self.labels else AcousticLabel.MUSIC
        return AcousticResult(
            label=label,
            window_start_utc=window.window_start_utc,
            music_prob=self.music_prob,
            speech_prob=self.speech_prob if label is AcousticLabel.MUSIC else 0.8,
            other_prob=self.other_prob,
            top_classes=[],
        )


@dataclass
class SequenceShazam:
    results: list[ShazamResult]
    calls: int = 0

    def __call__(self, window: AudioWindow) -> ShazamResult:
        self.calls += 1
        if not self.results:
            return ShazamResult(
                status=ShazamStatus.no_match,
                window_start_utc=window.window_start_utc,
            )
        result = self.results.pop(0)
        result.window_start_utc = window.window_start_utc
        return result


def _tier3(category: BroadcastCategory, *, brand: str | None = None, transcript: str = "...", sig: CommercialSignature | None = None):
    def _fn(window: AudioWindow) -> SpeechPipelineResult:
        return SpeechPipelineResult(
            window_start_utc=window.window_start_utc,
            status=SpeechPipelineStatus.ok,
            transcript=transcript,
            category=category,
            brand=brand,
            brand_mentions=[BrandMention(brand, "paid_ad")] if brand else [],
            commercial_signature=sig,
            confidence=0.9,
            rationale="...",
        )
    return _fn


def test_tier1_match_short_circuits(tmp_path: Path) -> None:
    store = BroadcastStore(tmp_path / "rc.db")
    try:
        orch = FunnelOrchestrator(
            tier1=FakeTier1(
                status=FingerprintStatus.match,
                artist="Taylor",
                title="Anti-Hero",
                track_id="Taylor - Anti-Hero.mp3",
            ),
            tier2=FakeTier2(label=AcousticLabel.SPEECH),  # would route to Tier 3 if reached
            tier3=_tier3(BroadcastCategory.DJ),
            resolver=None,
            store=store,
            suppress_singleton_fingerprint_matches=False,
        )
        r = orch.process(_make_window())
        assert r.stage is FunnelStage.tier1_fingerprint
        assert r.speech is None
        assert r.segment_input is not None
        assert r.segment_input.key.category is BroadcastCategory.SONG
        assert r.segment_input.key.song_id is not None  # upserted into songs table
    finally:
        store.close()


def test_singleton_fingerprint_match_is_suppressed() -> None:
    orch = FunnelOrchestrator(
        tier1=SequenceTier1(
            [
                FingerprintResult(
                    status=FingerprintStatus.match,
                    window_start_utc="",
                    artist="Nirvana",
                    title="Smells Like Teen Spirit",
                    track_id="data/reference/songs/Nirvana - Smells Like Teen Spirit.mp3",
                    match_score=15.0,
                )
            ]
        ),
        tier2=FakeTier2(
            label=AcousticLabel.MUSIC,
            music_prob=0.88,
            speech_prob=0.01,
            other_prob=0.11,
        ),
        tier3=_tier3(BroadcastCategory.DJ),
        resolver=None,
        store=None,
    )

    r = orch.process(_make_window())

    assert r.stage is FunnelStage.tier2_unknown_song
    assert r.fingerprint is not None
    assert r.fingerprint.status is FingerprintStatus.no_match
    assert "suppressed singleton fingerprint match" in (r.fingerprint.message or "")
    assert r.segment_input is not None
    assert r.segment_input.key.category is BroadcastCategory.SONG
    assert r.segment_input.key.song_id is None


def test_adjacent_same_track_fingerprint_match_is_confirmed() -> None:
    track_id = "data/reference/songs/The Cranberries - Zombie.webm"
    orch = FunnelOrchestrator(
        tier1=SequenceTier1(
            [
                FingerprintResult(
                    status=FingerprintStatus.match,
                    window_start_utc="",
                    artist="The Cranberries",
                    title="Zombie",
                    track_id=track_id,
                    match_score=51.0,
                ),
                FingerprintResult(
                    status=FingerprintStatus.match,
                    window_start_utc="",
                    artist="The Cranberries",
                    title="Zombie",
                    track_id=track_id,
                    match_score=111.0,
                ),
            ]
        ),
        tier2=FakeTier2(
            label=AcousticLabel.MUSIC,
            music_prob=0.88,
            speech_prob=0.01,
            other_prob=0.11,
        ),
        tier3=_tier3(BroadcastCategory.DJ),
        resolver=None,
        store=None,
    )

    first = orch.process(_make_window("2020-01-01T00:00:00.000Z"))
    second = orch.process(_make_window("2020-01-01T00:00:10.000Z"))

    assert first.stage is FunnelStage.tier2_unknown_song
    assert first.fingerprint is not None
    assert first.fingerprint.status is FingerprintStatus.no_match
    assert second.stage is FunnelStage.tier1_fingerprint
    assert second.fingerprint is not None
    assert second.fingerprint.status is FingerprintStatus.match
    assert second.segment_input is not None
    assert second.segment_input.key.category is BroadcastCategory.SONG


def test_low_confidence_fingerprint_match_requires_three_adjacent_hits() -> None:
    """Borderline audfprint candidates need stronger temporal support.

    The candidate floor can be lower than the strong acceptance score to
    recover weak real matches, but accepting a 45-59 score after only two
    windows would re-open the old false-positive cluster. Require three
    adjacent same-track hits until the current score reaches the strong floor.
    """
    track_id = "data/reference/songs/Linkin Park - Numb.mp3"
    orch = FunnelOrchestrator(
        tier1=SequenceTier1(
            [
                FingerprintResult(
                    status=FingerprintStatus.match,
                    window_start_utc="",
                    artist="Linkin Park",
                    title="Numb",
                    track_id=track_id,
                    match_score=49.0,
                ),
                FingerprintResult(
                    status=FingerprintStatus.match,
                    window_start_utc="",
                    artist="Linkin Park",
                    title="Numb",
                    track_id=track_id,
                    match_score=52.0,
                ),
                FingerprintResult(
                    status=FingerprintStatus.match,
                    window_start_utc="",
                    artist="Linkin Park",
                    title="Numb",
                    track_id=track_id,
                    match_score=55.0,
                ),
            ]
        ),
        tier2=FakeTier2(
            label=AcousticLabel.MUSIC,
            music_prob=0.88,
            speech_prob=0.01,
            other_prob=0.11,
        ),
        tier3=_tier3(BroadcastCategory.DJ),
        resolver=None,
        store=None,
    )

    first = orch.process(_make_window("2020-01-01T00:00:00.000Z"))
    second = orch.process(_make_window("2020-01-01T00:00:10.000Z"))
    third = orch.process(_make_window("2020-01-01T00:00:20.000Z"))

    assert first.stage is FunnelStage.tier2_unknown_song
    assert second.stage is FunnelStage.tier2_unknown_song
    assert second.fingerprint is not None
    assert "suppressed low-confidence fingerprint match" in (second.fingerprint.message or "")
    assert third.stage is FunnelStage.tier1_fingerprint
    assert third.fingerprint is not None
    assert third.fingerprint.status is FingerprintStatus.match


def test_adjacent_different_track_fingerprint_match_is_suppressed() -> None:
    orch = FunnelOrchestrator(
        tier1=SequenceTier1(
            [
                FingerprintResult(
                    status=FingerprintStatus.match,
                    window_start_utc="",
                    artist="Nirvana",
                    title="Smells Like Teen Spirit",
                    track_id="data/reference/songs/Nirvana - Smells Like Teen Spirit.mp3",
                    match_score=15.0,
                ),
                FingerprintResult(
                    status=FingerprintStatus.match,
                    window_start_utc="",
                    artist="The Cranberries",
                    title="Zombie",
                    track_id="data/reference/songs/The Cranberries - Zombie.webm",
                    match_score=51.0,
                ),
            ]
        ),
        tier2=FakeTier2(
            label=AcousticLabel.MUSIC,
            music_prob=0.88,
            speech_prob=0.01,
            other_prob=0.11,
        ),
        tier3=_tier3(BroadcastCategory.DJ),
        resolver=None,
        store=None,
    )

    first = orch.process(_make_window("2020-01-01T00:00:00.000Z"))
    second = orch.process(_make_window("2020-01-01T00:00:10.000Z"))

    assert first.stage is FunnelStage.tier2_unknown_song
    assert second.stage is FunnelStage.tier2_unknown_song
    assert second.fingerprint is not None
    assert second.fingerprint.status is FingerprintStatus.no_match
    assert "suppressed singleton fingerprint match" in (second.fingerprint.message or "")


def test_tier2_music_falls_through_to_unknown_song(tmp_path: Path) -> None:
    orch = FunnelOrchestrator(
        tier1=FakeTier1(status=FingerprintStatus.no_match),
        # Realistic pure-music probabilities: speech_prob must stay below the
        # speech-override threshold so the MUSIC short-circuit engages.
        tier2=FakeTier2(
            label=AcousticLabel.MUSIC, music_prob=0.9, speech_prob=0.01, other_prob=0.09
        ),
        tier3=_tier3(BroadcastCategory.DJ),  # should NOT be called
        resolver=None,
        store=None,
    )
    r = orch.process(_make_window())
    assert r.stage is FunnelStage.tier2_unknown_song
    assert r.speech is None  # Tier 3 was skipped


def test_shazam_fallback_used_when_music_and_enabled(tmp_path: Path) -> None:
    def shazam(window):
        return ShazamResult(
            status=ShazamStatus.match,
            window_start_utc=window.window_start_utc,
            artist="Some Artist",
            title="Some Title",
        )

    store = BroadcastStore(tmp_path / "rc.db")
    try:
        orch = FunnelOrchestrator(
            tier1=FakeTier1(status=FingerprintStatus.no_match),
            # Pure-music probabilities so the MUSIC short-circuit + Shazam
            # fallback engages instead of the new speech override.
            tier2=FakeTier2(
                label=AcousticLabel.MUSIC, music_prob=0.9, speech_prob=0.01, other_prob=0.09
            ),
            tier3=_tier3(BroadcastCategory.DJ),
            resolver=None,
            store=store,
            shazam_fn=shazam,
        )
        r = orch.process(_make_window())
        assert r.stage is FunnelStage.shazam_fallback
        assert r.segment_input is not None
        assert r.segment_input.key.category is BroadcastCategory.SONG
        assert r.segment_input.key.song_id is not None
    finally:
        store.close()


def test_shazam_fallback_reuses_cached_match_between_rechecks() -> None:
    shazam = SequenceShazam(
        [
            ShazamResult(
                status=ShazamStatus.match,
                window_start_utc="",
                artist="Djo",
                title="End of Beginning",
            ),
            ShazamResult(
                status=ShazamStatus.match,
                window_start_utc="",
                artist="Djo",
                title="End of Beginning",
            ),
        ]
    )
    orch = FunnelOrchestrator(
        tier1=FakeTier1(status=FingerprintStatus.no_match),
        tier2=FakeTier2(
            label=AcousticLabel.MUSIC,
            music_prob=0.9,
            speech_prob=0.01,
            other_prob=0.09,
        ),
        tier3=_tier3(BroadcastCategory.DJ),
        resolver=None,
        store=None,
        shazam_fn=shazam,
        shazam_recheck_windows=4,
    )

    results = [
        orch.process(_make_window(f"2020-01-01T00:00:{i * 10:02d}.000Z"))
        for i in range(6)
    ]

    assert shazam.calls == 2  # first window, then the fourth consecutive unknown window
    assert all(r.stage is FunnelStage.shazam_fallback for r in results)
    assert all(r.shazam is not None and r.shazam.title == "End of Beginning" for r in results)
    assert results[1].shazam is not None
    assert results[1].shazam.message == "cached shazam match"


def test_shazam_recheck_splits_back_to_back_unknown_songs() -> None:
    shazam = SequenceShazam(
        [
            ShazamResult(
                status=ShazamStatus.match,
                window_start_utc="",
                artist="Djo",
                title="End of Beginning",
            ),
            ShazamResult(
                status=ShazamStatus.match,
                window_start_utc="",
                artist="The Smashing Pumpkins",
                title="1979",
            ),
        ]
    )
    orch = FunnelOrchestrator(
        tier1=FakeTier1(status=FingerprintStatus.no_match),
        tier2=FakeTier2(
            label=AcousticLabel.MUSIC,
            music_prob=0.9,
            speech_prob=0.01,
            other_prob=0.09,
        ),
        tier3=_tier3(BroadcastCategory.DJ),
        resolver=None,
        store=None,
        shazam_fn=shazam,
        shazam_recheck_windows=2,
    )

    first = orch.process(_make_window("2020-01-01T00:00:00.000Z"))
    second = orch.process(_make_window("2020-01-01T00:00:10.000Z"))
    third = orch.process(_make_window("2020-01-01T00:00:20.000Z"))
    fourth = orch.process(_make_window("2020-01-01T00:00:30.000Z"))

    assert shazam.calls == 2
    assert first.segment_input is not None
    assert second.segment_input is not None
    assert third.segment_input is not None
    assert fourth.segment_input is not None
    assert first.segment_input.track_title == "End of Beginning"
    assert second.segment_input.track_title == "End of Beginning"
    assert third.segment_input.track_title == "1979"
    assert fourth.segment_input.track_title == "1979"
    assert first.segment_input.key != third.segment_input.key
    assert third.shazam is not None
    assert third.shazam.message is None  # actual recheck, not cached
    assert fourth.shazam is not None
    assert fourth.shazam.message == "cached shazam match"


def test_shazam_unknown_music_run_resets_after_dj_chatter() -> None:
    shazam = SequenceShazam(
        [
            ShazamResult(
                status=ShazamStatus.match,
                window_start_utc="",
                artist="Djo",
                title="End of Beginning",
            ),
            ShazamResult(
                status=ShazamStatus.match,
                window_start_utc="",
                artist="Djo",
                title="End of Beginning",
            ),
        ]
    )
    orch = FunnelOrchestrator(
        tier1=FakeTier1(status=FingerprintStatus.no_match),
        tier2=SequenceTier2(
            [
                AcousticLabel.MUSIC,
                AcousticLabel.SPEECH,
                AcousticLabel.MUSIC,
            ]
        ),
        tier3=_tier3(BroadcastCategory.DJ),
        resolver=None,
        store=None,
        shazam_fn=shazam,
        shazam_recheck_windows=4,
    )

    first = orch.process(_make_window("2020-01-01T00:00:00.000Z"))
    second = orch.process(_make_window("2020-01-01T00:00:10.000Z"))
    third = orch.process(_make_window("2020-01-01T00:00:20.000Z"))

    assert first.stage is FunnelStage.shazam_fallback
    assert second.stage is FunnelStage.tier3_speech
    assert third.stage is FunnelStage.shazam_fallback
    assert shazam.calls == 2  # the SPEECH window resets the unknown-music run
    assert third.shazam is not None
    assert third.shazam.message is None


def test_tier3_commercial_resolves_with_resolver(tmp_path: Path) -> None:
    store = BroadcastStore(tmp_path / "rc.db")
    try:
        resolver = CommercialIdentityResolver(store=store)
        sig = CommercialSignature(
            key_phrases=["save fifteen percent"], duration_bucket_seconds=15
        )
        orch = FunnelOrchestrator(
            tier1=FakeTier1(),
            tier2=FakeTier2(label=AcousticLabel.SPEECH),
            tier3=_tier3(
                BroadcastCategory.COMMERCIAL,
                brand="Geico",
                transcript="Save fifteen percent on car insurance",
                sig=sig,
            ),
            resolver=resolver,
            store=store,
            window_seconds=15.0,
        )
        r = orch.process(_make_window())
        assert r.stage is FunnelStage.tier3_speech
        assert r.commercial_resolution is not None
        assert r.commercial_resolution.commercial_id is not None
        assert r.segment_input is not None
        assert r.segment_input.key.category is BroadcastCategory.COMMERCIAL
        assert r.segment_input.key.commercial_id == r.commercial_resolution.commercial_id
        assert r.segment_input.brand_name == "Geico"
        assert r.brand_mentions and r.brand_mentions[0].name == "Geico"
    finally:
        store.close()


def test_tier3_dj_does_not_call_resolver(tmp_path: Path) -> None:
    store = BroadcastStore(tmp_path / "rc.db")
    try:
        # Resolver is non-None but should not be touched for non-COMMERCIAL classes.
        resolver = CommercialIdentityResolver(store=store)
        orch = FunnelOrchestrator(
            tier1=FakeTier1(),
            tier2=FakeTier2(label=AcousticLabel.SPEECH),
            tier3=_tier3(BroadcastCategory.DJ),
            resolver=resolver,
            store=store,
        )
        r = orch.process(_make_window())
        assert r.commercial_resolution is None
        assert r.segment_input is not None
        assert r.segment_input.key.category is BroadcastCategory.DJ
        # No commercial row was created.
        n = store.connection.execute("SELECT COUNT(*) FROM commercials").fetchone()[0]
        assert n == 0
    finally:
        store.close()


def test_speech_override_routes_music_window_to_tier3() -> None:
    """A MUSIC-labeled window with elevated speech_prob (DJ over music bed,
    or station sweep with voiceover) must go to Tier 3, not be short-circuited
    as ``tier2_unknown_song``.

    Probabilities mirror the live capture at 2026-05-25T08:37:51Z where a
    Live 105 station sweep produced ``m=0.43, s=0.22`` and our v1 funnel
    silently skipped it.
    """
    orch = FunnelOrchestrator(
        tier1=FakeTier1(status=FingerprintStatus.no_match),
        tier2=FakeTier2(
            label=AcousticLabel.MUSIC,
            music_prob=0.43,
            speech_prob=0.22,
            other_prob=0.35,
        ),
        tier3=_tier3(BroadcastCategory.STATION),
        resolver=None,
        store=None,
    )
    r = orch.process(_make_window())
    assert r.stage is FunnelStage.tier3_speech
    assert r.speech is not None
    assert r.segment_input is not None
    assert r.segment_input.key.category is BroadcastCategory.STATION


def test_speech_override_threshold_respects_configuration() -> None:
    """When ``speech_override_threshold`` is raised above the observed
    speech_prob, the MUSIC short-circuit re-engages. Validates the knob is wired.
    """
    orch = FunnelOrchestrator(
        tier1=FakeTier1(status=FingerprintStatus.no_match),
        tier2=FakeTier2(
            label=AcousticLabel.MUSIC,
            music_prob=0.43,
            speech_prob=0.22,
            other_prob=0.35,
        ),
        tier3=_tier3(BroadcastCategory.STATION),
        resolver=None,
        store=None,
        speech_override_threshold=0.30,
    )
    r = orch.process(_make_window())
    assert r.stage is FunnelStage.tier2_unknown_song
    assert r.speech is None


def test_speech_override_does_not_trigger_on_pure_music() -> None:
    """Pure-music windows on FM produce speech_prob ~0.00–0.05. They must
    still short-circuit at Tier 2 to keep Whisper off the music stream.
    """
    orch = FunnelOrchestrator(
        tier1=FakeTier1(status=FingerprintStatus.no_match),
        tier2=FakeTier2(
            label=AcousticLabel.MUSIC,
            music_prob=0.92,
            speech_prob=0.01,
            other_prob=0.07,
        ),
        tier3=_tier3(BroadcastCategory.DJ),  # would fire if routing broke
        resolver=None,
        store=None,
    )
    r = orch.process(_make_window())
    assert r.stage is FunnelStage.tier2_unknown_song
    assert r.speech is None


def test_no_tier3_when_disabled() -> None:
    orch = FunnelOrchestrator(
        tier1=FakeTier1(),
        tier2=FakeTier2(label=AcousticLabel.SPEECH),
        tier3=None,
        resolver=None,
        store=None,
    )
    r = orch.process(_make_window())
    assert r.stage is FunnelStage.skipped
    assert r.segment_input is None
