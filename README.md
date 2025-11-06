# WRP (Python) — 🚧 WORK IN PROGRESS

> **Status:** Early architecture drop to showcase the Project.  
> Full package/docs will land in the next few weeks.

## What is WRP?
WRP is a **protocol** and **runtime** for creating and sharing **AI workflows** that can run on any system.  
It is inspired by (and conceptually adjacent to) the MCP protocol for agent tools—but instead of “tools,” WRP
focuses on **workflows**. This lets authors build AI applications on top of any agentic SDK (OpenAI, custom,
etc.) and expose those workflows in a portable, client-consumable way.

**Goals**
- **Privacy-first:** enable workflows to run locally on users’ machines where possible. (or host your own Server)
- **Portability:** Clients can consume workflows from different providers.
- **Open ecosystem:** a foundation for community or monetizable marketplaces of reusable workflows.

**High-level components**
- **WRP Server:** where agentic logic/workflows live (runtime, settings, stores, telemetry).
- **WRP Client:** what end users/agents use to discover, run, and interact with workflows.
- **Runtime services:** redaction, session management, pluggable stores, telemetry spans/payloads,
  and robust client-server messaging (taken from MCP).

## Highlights
Server/runtime for a workflow & telemetry system with:
- pluggable Stores (memory / SQLite / Postgres)
- conversation seeding & channelized message history
- telemetry spans + encrypted span payloads
- workflow manager with per-workflow settings + persistence

## Composability: mix & match workflows from many providers
WRP is designed so **authors** can publish self-contained workflows that you can wire together with other workflows. 
Each workflow advertises a **typed input/output contract** and a **settings schema**, making
it safe to compose across organizational or runtime boundaries.

**Example: “AI Engineer” app assembled from multiple providers**
- **Provider A — `dev`**: Development workflow (turns tickets into code + reports)
- **Provider B — `research`**: Deep research workflow (sources, notes, citations)
- **Provider C — `computer`**: Computer/browser control workflow
- **Provider D — `test`**: Testing workflow
- **Provider E — `qa`**: QA workflow (acceptance criteria, triage, punch list)
- **Provider F — `design`**: Design workflow (UX notes, assets, diffs)

**Pseudo flow (illustrative, client API may differ)**
```python
# Pseudo-code: composing across providers A..F
ticket = "Implement secure workflow settings + persistence"

dev_out = await clientA.run_workflow("dev", {"prompt": ticket})
research_out = await clientB.run_workflow("research", {"topic": dev_out["developer_report"]})
nav_out = await clientC.run_workflow("computer", {"queries": research_out["sources"]})
test_out = await clientD.run_workflow("test", {
    "prompt": "write tests for the new settings persistence",
    "repair_code": True
})
qa_out = await clientE.run_workflow("qa", {"artifact": dev_out["developer_report"], "notes": test_out["test_report"]})
design_out = await clientF.run_workflow("design", {"brief": qa_out["punch_list"]})
```

**Why this matters**
- **Best-of-breed**: Pick the strongest workflow for each stage, from any vendor or open-source author.
- **Replaceable parts**: Swap a workflow without rewriting the rest of the app (contracts stay stable).
- **Data control**: Keep sensitive steps on-device; send only sanitized outputs to remote providers.
- **Traceability**: End-to-end spans give you provenance, timing, and usage across the whole pipeline.

**Authoring for composability (what to export)**
- **Typed contracts**: `WorkflowInput` / `WorkflowOutput` models that are minimal and explicit.
- **Settings**: `WorkflowSettings` with sensible defaults, plus `locked` fields if needed.
- **Channels**: Use channelized conversations to keep seeds, debug, and user-visible output separate.
- **Telemetry**: Emit spans (start/end/points); keep sensitive data in encrypted payloads only.

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
python examples/ai_engineer_server.py
```

## Status
Work-in-progress. This repo is being pushed early; active development continues.

## License
© 2025 Iheb Salah — Licensed under MIT. — see LICENSE.