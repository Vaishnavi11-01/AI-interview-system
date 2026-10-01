# Sample run artifacts

This folder contains genuine artifacts from processing the supplied sample video:

- `data/visitors.sqlite3`: SQLite visitor embeddings and event rows.
- `logs/events.log`: system, registration, tracking, entry, and exit records.
- `logs/entries/2026-10-01/`: two timestamped entry snapshots.
- `logs/exits/2026-10-01/`: two timestamped exit snapshots.

This run recorded 2 visitor IDs and 4 event rows (one entry and one exit per observed track). In this evidence copy, event image paths are relative to the repository root and point to the packaged snapshots. Run `python main.py --count` to see the count in the active database. For a new run, copy its generated files into this folder again if you want to replace this evidence.