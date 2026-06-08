#!/usr/bin/env python3
"""
Re-map orphaned/mis-mapped Cursor chats back to the radio-classifier workspace.

WHY: After the crash, the main project chats are bound to the wrong workspace:
  - "Local terrestrial radio classifier project" -> /home/eamon/dev (parent folder)
  - several others             -> workspaceIdentifier = null  ("disconnected")
Cursor's history panel filters by workspaceIdentifier.id, so they don't show up
in the /home/eamon/dev/radio-classifier window. This rewrites their
workspaceIdentifier to the radio-classifier workspace so they reappear.

SAFETY:
  * Cursor MUST be fully closed before running (File > Exit / quit the app,
    not just close the window). The script takes an EXCLUSIVE sqlite lock and
    aborts if Cursor still holds the database.
  * A timestamped backup of state.vscdb is made before any write.
  * Default is a DRY RUN. Re-run with --apply to actually write.

USAGE (from a normal WSL terminal, NOT from inside Cursor):
    python3 fix_chat_mapping.py          # dry run, shows what would change
    python3 fix_chat_mapping.py --apply  # perform the remap
"""
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

GLOBAL_DB = Path(
    "/mnt/c/Users/eamon/AppData/Roaming/Cursor/User/globalStorage/state.vscdb"
)

# Target workspace = /home/eamon/dev/radio-classifier (hash f0ad...).
# Mirrors the exact shape Cursor uses for a working WSL-remote chat.
RADIO_WS_ID = "f0ad704965881269057bc55899e9e77b"
RADIO_WORKSPACE_IDENTIFIER = {
    "id": RADIO_WS_ID,
    "uri": {
        "$mid": 1,
        "fsPath": "\\home\\eamon\\dev\\radio-classifier",
        "_sep": 1,
        "external": "vscode-remote://wsl%2Bubuntu/home/eamon/dev/radio-classifier",
        "path": "/home/eamon/dev/radio-classifier",
        "scheme": "vscode-remote",
        "authority": "wsl+ubuntu",
    },
}

# Chats to re-map to radio-classifier. Edit this list if you disagree.
TARGET_COMPOSER_IDS = [
    "65258b42-e176-4766-8c61-0da51784d0b8",  # Local terrestrial radio classifier project (1137 msgs)
    "7c5cb2c1-0ff3-44d1-85ae-b9abe0703cee",  # Audio classification pipeline specification (37 msgs)
    "4dd050e8-69dd-4966-98a8-731b6e7c95bc",  # SQLite database schema and init script (1000 msgs)
    "060d98d6-3f09-4087-9676-df9bad2fefc5",  # Previous conversation context (11 msgs)
]


def main() -> int:
    apply = "--apply" in sys.argv

    if not GLOBAL_DB.exists():
        print(f"ERROR: database not found: {GLOBAL_DB}")
        return 1

    if apply:
        backup = GLOBAL_DB.with_suffix(f".vscdb.recovery-bak-{int(time.time())}")
        shutil.copy2(GLOBAL_DB, backup)
        print(f"Backup written: {backup}")

    # Connect. If Cursor is still running it holds a lock; EXCLUSIVE will fail.
    con = sqlite3.connect(str(GLOBAL_DB), timeout=2.0)
    con.isolation_level = None
    cur = con.cursor()
    if apply:
        try:
            cur.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as exc:
            print(f"\nERROR: database is locked ({exc}).")
            print("Cursor is still running. Fully quit Cursor and retry.")
            con.close()
            return 2

    mode = "APPLY" if apply else "DRY RUN"
    print(f"\n=== {mode}: remapping {len(TARGET_COMPOSER_IDS)} chats -> {RADIO_WS_ID} ===\n")

    changed = 0
    for cid in TARGET_COMPOSER_IDS:
        cur.execute(
            "SELECT value FROM cursorDiskKV WHERE key=?", (f"composerData:{cid}",)
        )
        row = cur.fetchone()
        if row is None:
            print(f"  SKIP {cid}: not found")
            continue
        data = json.loads(row[0])
        before = data.get("workspaceIdentifier")
        before_id = before.get("id") if isinstance(before, dict) else before
        if before_id == RADIO_WS_ID:
            print(f"  OK   {cid}: already mapped to radio-classifier; skipping")
            continue
        print(f"  MAP  {cid} | {data.get('name')!r}")
        print(f"         from: {before_id}")
        print(f"         to:   {RADIO_WS_ID}")
        if apply:
            data["workspaceIdentifier"] = RADIO_WORKSPACE_IDENTIFIER
            cur.execute(
                "UPDATE cursorDiskKV SET value=? WHERE key=?",
                (json.dumps(data), f"composerData:{cid}"),
            )
            changed += 1

    if apply:
        cur.execute("COMMIT")
        print(f"\nDone. {changed} chat(s) remapped. Restart Cursor in the "
              f"radio-classifier window and check chat history.")
    else:
        print("\nDry run only — no changes written. Re-run with --apply (Cursor closed).")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
