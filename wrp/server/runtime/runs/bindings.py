# wrp/server/runtime/runs/bindings.py
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from wrp.server.runtime.exceptions import RunStateError, WorkflowMarkedError, WorkflowMarkedFailure

from wrp.server.runtime.conversations.seeding import ConversationSeeding, SeedingRunFilter, default_conversation_seeding
from .types import RunOutcome, RunState
from ..telemetry.service import RunTelemetryService
from wrp.server.runtime.conversations.service import ConversationsService


class RunBindings(BaseModel):
    """Bound into ctx.run for the duration of a workflow run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    workflow_name: str
    thread_id: str | None = None

    conversation_seeding: ConversationSeeding = Field(default_factory=default_conversation_seeding)
    seeding_run_filter: SeedingRunFilter = Field(default_factory=SeedingRunFilter)

    conversations: ConversationsService
    telemetry: RunTelemetryService

    # ----- developer-controlled outcomes -----
    async def conclude_failure(self, reason: str, *, output: dict | Any | None = None) -> None:
        """
        Mark the run as 'failed' (business failure), persist optional structured output,
        then raise an internal signal to unwind the workflow.
        """
        sid = self.conversations._run.system_session_id
        meta = await self.conversations._store.get_run(sid, self.run_id)
        if meta and meta.state == RunState.concluded:
            raise RunStateError("Run already concluded")
        # coerce output if it's a pydantic model
        if output is not None and hasattr(output, "model_dump"):
            output = output.model_dump()  # type: ignore[attr-defined]
        await self.conversations._store.conclude_run(
            sid, self.run_id, RunOutcome.failed, error=reason, run_output=output
        )
        raise WorkflowMarkedFailure(reason)

    async def conclude_error(self, reason: str, *, output: dict | None = None) -> None:
        """
        Mark the run as 'error' (technical fault), persist optional structured details,
        then raise an internal signal to unwind the workflow.
        """
        sid = self.conversations._run.system_session_id
        meta = await self.conversations._store.get_run(sid, self.run_id)
        if meta and meta.state == RunState.concluded:
            raise RunStateError("Run already concluded")
        await self.conversations._store.conclude_run(sid, self.run_id, RunOutcome.error, error=reason, run_output=output)
        raise WorkflowMarkedError(reason)