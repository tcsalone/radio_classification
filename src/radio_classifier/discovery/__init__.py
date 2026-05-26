"""Shazam discovery workflow — list discovered songs and promote them to the
local fingerprint tracklist.

Pure SQLite reads + a single file append; no network, no model loads.
"""

from radio_classifier.discovery.songs import (
    DiscoveryRow,
    PromoteResult,
    PromotedTrack,
    list_shazam_discoveries,
    promote_to_tracklist,
)

__all__ = [
    "DiscoveryRow",
    "PromoteResult",
    "PromotedTrack",
    "list_shazam_discoveries",
    "promote_to_tracklist",
]
