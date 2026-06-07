"""Shazam discovery workflow — list discovered songs and promote them to the
local fingerprint tracklist.

Pure SQLite reads + a single file append; no network, no model loads.
"""

from radio_classifier.discovery.songs import (
    DedupeGroup,
    DedupeReport,
    DiscoveryRow,
    PromoteResult,
    PromotedTrack,
    dedupe_songs,
    list_shazam_discoveries,
    promote_to_tracklist,
)
from radio_classifier.discovery.stitch import (
    StitchGroup,
    StitchReport,
    stitch_song_plays,
)

__all__ = [
    "DedupeGroup",
    "DedupeReport",
    "DiscoveryRow",
    "PromoteResult",
    "PromotedTrack",
    "StitchGroup",
    "StitchReport",
    "dedupe_songs",
    "list_shazam_discoveries",
    "promote_to_tracklist",
    "stitch_song_plays",
]
