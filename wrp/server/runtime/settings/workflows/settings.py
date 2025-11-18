# wrp/server/runtime/settings/workflows/settings.py
from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, PrivateAttr


class WorkflowSettings(BaseModel):
    """
    Base class for author/user-configurable workflow settings (not per-run).
    Subclass this per workflow. Extra fields are allowed to keep it flexible.
    """

    model_config = ConfigDict(extra="allow")

    # Internal, instance-scoped flag; manager also exposes a canonical view.
    _override_status: bool = PrivateAttr(default=False)

    # Field-level locks (declare in subclasses: e.g., `locked = {"model"}`)
    locked: ClassVar[set[str]] = set()

    def __getattr__(self, name: str) -> Any:
        """
        Allow workflow authors to define arbitrary settings fields without
        triggering static type-analysis errors while still failing fast at runtime.
        """
        raise AttributeError(f"{self.__class__.__name__!s} has no attribute {name!r}")

    def settings_overridden(
        self,
        *,
        ctx: "Context | None" = None,  # type: ignore[name-defined]
        workflow: str | None = None,
    ) -> bool:
        """
        True if this settings object reflects an explicit client override.
        If a Context is provided, prefers the manager's canonical flag.
        """
        if ctx is not None:
            wf_name = workflow
            if wf_name is None:
                # Avoid the public .run property (it raises outside a workflow run).
                run_bindings = getattr(ctx, "_run", None)
                if run_bindings is not None:
                    wf_name = getattr(run_bindings, "workflow_name", None)
            if wf_name:
                # canonical flag from the workflow manager
                return bool(ctx.wrp._workflow_manager.settings_overridden(wf_name))
        return bool(self._override_status)
