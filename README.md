# WRP (Python)

Server/runtime for a workflow & telemetry system with:
- pluggable Stores (memory / SQLite / Postgres)
- conversation seeding & channelized message history
- telemetry spans + encrypted span payloads
- workflow manager with per-workflow settings + persistence

## Layout
wrp-python/
wrp/ # Python package: client/server/shared

bash
Copy code

## Quick start
```bash
python -m pip install -e .
```bash

Status
Work-in-progress. Pushing early to showcase architecture & complexity.