"""Optional Shazam fallback for unknown-song identification (network)."""

from radio_classifier.music.shazam_client import identify_window_sync
from radio_classifier.music.types import ShazamResult, ShazamStatus

__all__ = ["ShazamResult", "ShazamStatus", "identify_window_sync"]
