"""Seeding toolchain (tracklist scrape, yt-dlp downloads, eval harness).

Optional ``[seeding]`` extra; never imported by the runtime ingest path.
"""

from radio_classifier.seeding.eval import EvalReport, EvalRow, evaluate, load_truth
from radio_classifier.seeding.scrape import Track, dedupe_tracks, fetch_html, parse_tracklist

__all__ = [
    "EvalReport",
    "EvalRow",
    "Track",
    "dedupe_tracks",
    "evaluate",
    "fetch_html",
    "load_truth",
    "parse_tracklist",
]
