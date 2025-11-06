# README.md

# WRP (Python) — 🚧 WORK IN PROGRESS

> **Status:** Early architecture drop to showcase the Project.  
> Full package/docs will land in the next few weeks.

Server/runtime for a workflow & telemetry system with:
- pluggable Stores (memory / SQLite / Postgres)
- conversation seeding & channelized message history
- telemetry spans + encrypted span payloads
- workflow manager with per-workflow settings + persistence

## Layout
```
wrp-python/
  wrp/                # Python package: client / server / shared
  examples/
    ai_engineer_server.py
  tests/
```

## Quick start
```bash
python -m pip install -e .
```

## Status
Work-in-progress. This repo is being pushed early; active development continues.