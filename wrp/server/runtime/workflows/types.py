# wrp/server/runtime/workflows/types.py
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mcp.types import Icon


class WorkflowInput(BaseModel):
    """Base class for all WRP workflow inputs.

    Developers should subclass this and declare their workflow's input fields.
    """

    model_config = ConfigDict(extra="allow")


class WorkflowOutput(BaseModel):
    """Base class for all WRP workflow outputs.

    Developers should subclass this and declare their workflow's output fields.
    """

    model_config = ConfigDict(extra="allow")


class WorkflowDescriptor(BaseModel):
    """Public metadata for a registered workflow."""

    name: str = Field(description="Unique workflow name.")
    title: str | None = Field(default=None, description="Human-readable title.")
    description: str | None = Field(default=None, description="What the workflow does.")
    inputSchema: dict[str, Any] = Field(description="JSON schema for WorkflowInput.")
    outputSchema: dict[str, Any] = Field(description="JSON schema for WorkflowOutput.")
    icons: list[Icon] | None = Field(default=None, description="Optional icon list.")
    settingsSchema: dict[str, Any] | None = Field(
        default=None,
        description="JSON schema for WorkflowSettings derived from the default instance.",
    )

    model_config = ConfigDict(extra="allow")


class RunWorkflowResult(BaseModel):
    """Standardized run result envelope for workflows."""

    output: dict[str, Any] | None = Field(
        default=None,
        description="WorkflowOutput serialized to a JSON-compatible dict (dumped).",
    )
    isError: bool = Field(default=False, description="True if the run failed.")
    error: str | None = Field(default=None, description="Error message when isError=True.")

    model_config = ConfigDict(extra="allow")