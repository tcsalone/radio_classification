-- radio-classifier — schema v3
-- Five-class broadcast events with brand attribution, song fingerprinting,
-- text-derived commercial identity, and capture-run provenance.
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('version', '3');

CREATE TABLE IF NOT EXISTS brands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name TEXT NOT NULL UNIQUE,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS songs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audfprint_track_id TEXT,
    artist TEXT,
    title TEXT,
    source TEXT NOT NULL DEFAULT 'audfprint'
        CHECK (source IN ('audfprint', 'shazam', 'manual')),
    first_seen_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (artist, title, source)
);

CREATE TABLE IF NOT EXISTS commercials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    brand_id INTEGER NOT NULL REFERENCES brands(id),
    duration_bucket_seconds INTEGER NOT NULL,
    minhash_hex TEXT NOT NULL,
    reference_transcript TEXT NOT NULL,
    first_heard_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    play_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE (brand_id, duration_bucket_seconds, minhash_hex)
);

CREATE TABLE IF NOT EXISTS capture_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    started_utc TEXT NOT NULL,
    ended_utc TEXT,
    pipeline_version TEXT NOT NULL,
    host TEXT,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS broadcast_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_start TEXT NOT NULL,
    timestamp_end TEXT,
    duration REAL,
    category TEXT NOT NULL CHECK (category IN ('SONG', 'DJ', 'COMMERCIAL', 'STATION', 'PSA_NEWS')),
    song_id INTEGER REFERENCES songs(id),
    commercial_id INTEGER REFERENCES commercials(id),
    brand_id INTEGER REFERENCES brands(id),
    artist TEXT,
    track_title TEXT,
    brand_name TEXT,
    transcript_excerpt TEXT,
    confidence REAL,
    capture_run_id INTEGER REFERENCES capture_runs(id),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS brand_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id INTEGER NOT NULL REFERENCES broadcast_events(id) ON DELETE CASCADE,
    brand_id INTEGER NOT NULL REFERENCES brands(id),
    mention_type TEXT NOT NULL CHECK (mention_type IN ('paid_ad', 'dj_shoutout', 'tag')),
    heard_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_start ON broadcast_events (timestamp_start);
CREATE INDEX IF NOT EXISTS idx_events_category ON broadcast_events (category);
CREATE INDEX IF NOT EXISTS idx_events_brand ON broadcast_events (brand_id);
CREATE INDEX IF NOT EXISTS idx_events_song ON broadcast_events (song_id);
CREATE INDEX IF NOT EXISTS idx_events_commercial ON broadcast_events (commercial_id);
CREATE INDEX IF NOT EXISTS idx_events_capture_run ON broadcast_events (capture_run_id);
CREATE INDEX IF NOT EXISTS idx_mentions_brand_time ON brand_mentions (brand_id, heard_utc);
CREATE INDEX IF NOT EXISTS idx_commercials_brand ON commercials (brand_id);
