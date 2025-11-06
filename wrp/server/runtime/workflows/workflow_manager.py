# wrp/server/runtime/workflows/workflow_manager.py
from __future__ import annotations as _annotations

from typing import Any, Callable, TYPE_CHECKING

from mcp.server.fastmcp.utilities.logging import get_logger

from wrp.server.runtime.exceptions import (
    WorkflowError,
    WorkflowMarkedError,
    WorkflowMarkedFailure,
)
from wrp.server.runtime.runs.bindings import RunBindings
from wrp.server.runtime.conversations.service import ConversationsService
from wrp.server.runtime.runs.requests.options import parse_run_request_options
from wrp.server.runtime.runs.types import RunSettings
from wrp.server.runtime.store.codecs import serializer
from wrp.server.runtime.telemetry.service import RunTelemetryService
from wrp.server.runtime.runs.types import RunMeta, RunOutcome
from wrp.server.runtime.workflows.base import Workflow
from wrp.server.runtime.conversations.seeding import WorkflowConversationSeeding
from wrp.shared.context import LifespanContextT, RequestT
from wrp.types import Icon
from .types import (
    RunWorkflowResult,
    WorkflowDescriptor,
    WorkflowInput,
    WorkflowOutput,
)
from .settings import WorkflowSettings
from wrp.server.runtime.store.base import Store

if TYPE_CHECKING:
    from wrp.server.session import ServerSessionT
    from wrp.server.runtime.server import Context

logger = get_logger(__name__)


class WorkflowManager:
    """Registry + invocation for WRP workflows."""

    def __init__(
        self,
        warn_on_duplicate_workflows: bool = True,
        store: Store | None = None,
    ):
        self._workflows: dict[str, Workflow] = {}
        self.warn_on_duplicate_workflows = warn_on_duplicate_workflows
        # current settings per workflow name (validated instances)
        self._settings: dict[str, WorkflowSettings] = {}
        self._settings_overridden: dict[str, bool] = {}
        self._store: Store | None = store

    def get(self, name: str) -> Workflow | None:
        """Get a workflow by name."""
        return self._workflows.get(name)

    def list_descriptors(self) -> list[WorkflowDescriptor]:
        """List all registered workflows (no pagination)."""
        return [wf.descriptor() for wf in self._workflows.values()]

    def add_workflow(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        input_model: type[WorkflowInput] | None = None,
        output_model: type[WorkflowOutput] | None = None,
        icons: list[Icon] | None = None,
        seeding: WorkflowConversationSeeding | None = None,
        input_limit_bytes: int | None = None,
        settings_default: WorkflowSettings | None = None,
        settings_allow_override: bool = True,
    ) -> Workflow:
        """Register a workflow function."""
        wf = Workflow.from_function(
            fn,
            name=name,
            title=title,
            description=description,
            input_model=input_model,
            output_model=output_model,
            icons=icons,
            seeding=seeding,
            input_limit_bytes=input_limit_bytes,
            settings_default=settings_default,
            settings_allow_override=settings_allow_override,
        )

        existing = self._workflows.get(wf.name)
        if existing:
            if self.warn_on_duplicate_workflows:
                logger.warning("Workflow already exists: %s", wf.name)
            return existing

        self._workflows[wf.name] = wf
        # initialize current settings (copy default)
        if wf.settings_default is not None:
            inst = wf.settings_default.copy(deep=True)
            # propagate initial flag to instance
            inst._override_status = False
            self._settings[wf.name] = inst
            self._settings_overridden[wf.name] = False
        return wf

    # -------- settings API (author/client configurable, non-run) ----------
    def get_settings(self, name: str) -> WorkflowSettings | None:
        """Get the current settings for a workflow.

        If settings have not been explicitly updated, this returns a copy of the
        default settings provided when the workflow was registered.

        Args:
            name: The name of the workflow.

        Returns:
            The current settings object, or None if the workflow does not support settings.
        """
        wf = self.get(name)
        if not wf or wf.settings_default is None:
            return None
        cur = self._settings.get(name)
        if cur is not None:
            # sync instance flag with manager view
            cur._override_status = bool(self._settings_overridden.get(name, False))
            return cur
        # fall back to default if somehow absent
        cur = wf.settings_default.copy(deep=True)
        cur._override_status = False
        self._settings[name] = cur
        self._settings_overridden[name] = False
        return cur

    async def update_settings(self, name: str, data: dict[str, Any]) -> WorkflowSettings:
        """Update the settings for a workflow.

        This performs a partial update, merging the provided data with the current settings.
        The merged data is then validated against the workflow's settings model.
        The updated settings are persisted to the configured store and cached in memory.

        This operation respects two constraints:
        1. The workflow must have `settings_allow_override=True`.
        2. Individual fields defined as `locked` within the settings model cannot be changed.

        Args:
            name: The name of the workflow.
            data: A dictionary containing the new settings values to merge.

        Returns:
            The updated and validated settings object.

        Raises:
            WorkflowError: If the workflow does not exist, does not support settings,
                         disallows overrides, or if an attempt is made to change a locked setting.
            pydantic.ValidationError: If the provided data is invalid for the settings model.
        """
        wf = self.get(name)
        if not wf or wf.settings_default is None:
            raise WorkflowError(f"Workflow '{name}' does not declare settings (no default provided)")
        if not wf.settings_allow_override:
            raise WorkflowError(f"Workflow '{name}' does not allow settings overrides")
        model = wf.settings_default.__class__
        # current + field locks enforcement
        current = self.get_settings(name)
        base_dict = current.model_dump() if current else {}
        locked = getattr(model, "locked", set()) or set()
        for k in data or {}:
            if k in locked and (k in base_dict) and data[k] != base_dict[k]:
                raise WorkflowError(f"Setting '{k}' is locked and cannot be overridden")
        merged = {**base_dict, **(data or {})}
        inst = model.model_validate(merged)
        # keep canonical flag in the instance
        inst = inst.model_copy(update={"_override_status": True})
        self._settings[name] = inst
        self._settings_overridden[name] = True
        # persist (best-effort)
        if self._store is not None:
            try:
                await self._store.upsert_workflow_settings(name, inst.model_dump(), overridden=True)
            except Exception:
                logger.exception("Failed to persist workflow settings for %s", name)
        return inst

    async def load_persisted_settings_if_needed(self, name: str) -> None:
        """Hydrate in-memory cache for a workflow from the store, if not already overridden in memory."""
        if self._store is None:
            return
        wf = self.get(name)
        if not wf or wf.settings_default is None:
            return
        # Already have an override in memory -> nothing to do
        if self._settings_overridden.get(name, False):
            return
        try:
            row = await self._store.get_workflow_settings(name)
        except Exception:
            logger.exception("Failed to load persisted settings for %s", name)
            return
        if not row:
            return
        values, overridden = row
        model = wf.settings_default.__class__
        try:
            inst = model.model_validate(values)
            inst = inst.model_copy(update={"_override_status": bool(overridden)})
            self._settings[name] = inst
            self._settings_overridden[name] = bool(overridden)
        except Exception:
            logger.exception("Persisted settings invalid for %s; ignoring", name)

    def settings_overridden(self, name: str) -> bool:
        """Check if a workflow's settings have been explicitly updated.

        Args:
            name: The name of the workflow.

        Returns:
            True if `update_settings` has been called for this workflow, False otherwise.
        """
        return bool(self._settings_overridden.get(name, False))

    async def run(
        self,
        name: str,
        input_dict: dict[str, Any],
        ctx: Context[ServerSessionT, LifespanContextT, RequestT] | None = None,
    ) -> RunWorkflowResult:
        """Execute a workflow by name, recording its execution as a persistent 'run'.

        This method orchestrates the entire lifecycle of a workflow execution:
        1. Parses run-specific options (e.g., `thread_id`) from the request.
        2. Creates a unique run record in the run store.
        3. Attaches run-specific APIs (`ctx.run.telemetry`, `ctx.run.conversations`) to the context.
        4. Executes the workflow function.
        5. Records the outcome (success or error) via telemetry and concludes the run.

        Args:
            name: The name of the workflow to run.
            input_dict: A dictionary containing the input data for the workflow.
            ctx: The server context, which is required for creating and managing the run.

        Returns:
            A standardized result object indicating success or failure.
        """
        wf = self.get(name)
        if not wf:
            return RunWorkflowResult(isError=True, error=f"Unknown workflow: {name}")

        # Require a Context to attach per-run APIs; runs are not supported outside of a request.
        if ctx is None:
            return RunWorkflowResult(isError=True, error="Context is required to run workflows")

        # Enforce input size limit, preferring the workflow-specific setting over the global default.
        eff_limit = (
            wf.input_limit_bytes
            if wf.input_limit_bytes is not None
            else ctx.wrp.settings.global_input_limit_bytes
        )
        if eff_limit is not None:
            raw_size = len(serializer.dumps(input_dict))
            if raw_size > eff_limit:
                return RunWorkflowResult(
                    isError=True,
                    error=f"Workflow input too large: {raw_size} bytes > limit {eff_limit} bytes",
                )

        # Parse caller-provided run options from the raw request data.
        req_data = getattr(ctx.request_context, "request", None)
        opts = parse_run_request_options(req_data)
        # NOTE: parse_run_request_options expects a dict-like object; the server preserves the original request data.
        # Effective seeding source: workflow-specific overrides global; caller meta can still override
        # unless 'deny_seeding' is set by the effective source.
        effective_seeding = wf.seeding or ctx.wrp.default_seeding
        # Merge with effective seeding (author overrides caller when deny=true; otherwise provide defaults)
        if effective_seeding and effective_seeding.deny_seeding:
            # hard block any cross-run seeding
            from wrp.server.runtime.conversations.seeding import ConversationSeedingNone, RunFilter as _RunFilter

            opts.conversation_seeding = ConversationSeedingNone()
            opts.run_filter = _RunFilter()
        elif (effective_seeding is not None) and (not opts.conversation_seeding_specified):
            # apply author's/global default only if caller didn't supply seeding settings
            opts.conversation_seeding = effective_seeding.default_seeding

        # Create the run record and persist it.
        store = ctx.wrp.store
        # Allocate simple 3-digit run id (001..999) within (workflow, thread) scope
        run_id = await store.alloc_run_id(wf.name, opts.thread_id)
        meta = RunMeta(run_id=run_id, workflow_name=wf.name, thread_id=opts.thread_id)
        await store.create_run(meta)

        # Bind author-facing façades into ctx.run, making them available to the workflow.
        # Channel defaults/limits from the effective seeding (workflow > server default)
        default_channels = (effective_seeding.default_channels if effective_seeding else ["default"])
        allowed_channels = (effective_seeding.allowed_channels if effective_seeding else None)

        conversations_service = ConversationsService(
            store,
            meta,
            conversation_seeding=opts.conversation_seeding,
            run_filter=opts.run_filter,
            default_channels=default_channels,
            allowed_channels=allowed_channels,
            on_update=ctx.wrp.notify_resource_updated,
        )
        telemetry_service = RunTelemetryService(
            store,
            meta,
            on_payload_update=ctx.wrp.notify_resource_updated,
        )
        bindings = RunBindings(
            run_id=run_id,
            workflow_name=wf.name,
            thread_id=opts.thread_id,
            conversation_seeding=opts.conversation_seeding,
            run_filter=opts.run_filter,
            conversations=conversations_service,
            telemetry=telemetry_service,
        )
        ctx._attach_run(bindings)

        # Build run settings for telemetry using the same effective source.
        default_channels = (effective_seeding.default_channels if effective_seeding else ["default"])
        allowed_channels = (effective_seeding.allowed_channels if effective_seeding else None)
        seeding_deny = (effective_seeding.deny_seeding if effective_seeding else False)

        run_settings = RunSettings(
            conversation_seeding=opts.conversation_seeding,
            run_filter=opts.run_filter,
            ignore_thread=opts.ignore_thread,
            default_channels=default_channels,
            allowed_channels=allowed_channels,
            seeding_deny=seeding_deny,
            conversation_seeding_specified=opts.conversation_seeding_specified,
            run_filter_specified=opts.run_filter_specified,
        )

        # run-level span to trace the entire workflow execution.
        run_span_id = await ctx.run.telemetry.run_start(
            name=wf.name,
            thread_id=opts.thread_id,
            settings=run_settings,
            workflow_name=wf.name,
            run_input=input_dict,
        )

        try:
            output_model = await wf.run(input_dict, ctx)
            output_dict = output_model.model_dump()

            # Conclude the run successfully. Output is captured as a telemetry payload, not on the run record itself.
            try:
                await store.conclude_run(run_id, RunOutcome.success, run_output=None)
            finally:
                await ctx.run.telemetry.run_end(
                    span_id=run_span_id,
                    status="ok",
                    outcome=RunOutcome.success,
                    workflow_name=wf.name,
                    output=output_dict,
                    error=None,
                )
            return RunWorkflowResult(output=output_dict)
        except WorkflowMarkedFailure as e:
            # Author marked run as a business failure via ctx.run.conclude_failure().
            # Run is already concluded, so just emit telemetry and return.
            try:
                await ctx.run.telemetry.run_end(
                    span_id=run_span_id,
                    status="error",
                    outcome=RunOutcome.failed,
                    workflow_name=wf.name,
                    output=None,
                    error=str(e),
                )
            except Exception:
                pass
            return RunWorkflowResult(isError=True, error=str(e))
        except WorkflowMarkedError as e:
            # Author marked run as a technical error via ctx.run.conclude_error().
            # Run is already concluded, so just emit telemetry and return.
            try:
                await ctx.run.telemetry.run_end(
                    span_id=run_span_id,
                    status="error",
                    outcome=RunOutcome.error,
                    workflow_name=wf.name,
                    output=None,
                    error=str(e),
                )
            except Exception:
                pass
            return RunWorkflowResult(isError=True, error=str(e))
        except WorkflowError as e:
            # Controlled, user-facing error; conclude here.
            await store.conclude_run(run_id, RunOutcome.error, error=str(e), run_output=None)
            try:
                await ctx.run.telemetry.run_end(
                    span_id=run_span_id,
                    status="error",
                    outcome=RunOutcome.error,
                    workflow_name=wf.name,
                    output=None,
                    error=str(e),
                )
            except Exception:
                pass
            return RunWorkflowResult(isError=True, error=str(e))
        except Exception as e:  # Defensive: unexpected crash
            logger.exception("Unexpected exception while running workflow '%s'", name)
            await store.conclude_run(run_id, RunOutcome.error, error=str(e), run_output=None)
            try:
                await ctx.run.telemetry.run_end(
                    span_id=run_span_id,
                    status="error",
                    outcome=RunOutcome.error,
                    workflow_name=wf.name,
                    output=None,
                    error=str(e),
                )
            except Exception:
                pass
            return RunWorkflowResult(isError=True, error=str(e))