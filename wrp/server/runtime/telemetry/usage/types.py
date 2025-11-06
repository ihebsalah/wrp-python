# wrp/server/runtime/telemetry/usage/types.py
from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field


class AggregationDiagnostics(BaseModel):
    """
    Diagnostics for agent-level usage aggregation.
    Emitted as human-readable annotations by the service.
    """
    missing_agent_id: bool = False
    window_missing: bool = False  # couldn't resolve agent window (start/end ts)
    ambiguous_agent_id: bool = False  # overlapping agent windows with the same agent_id
    untagged_in_window: int = 0  # count of LLM/tool events in window without agent_id
    included_llm_events: int = 0  # counted toward this agent
    considered_llm_events: int = 0  # LLM end events observed in window (any tagging)
    notes: List[str] = Field(default_factory=list)
