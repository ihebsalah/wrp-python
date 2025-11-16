# wrp/server/runtime/workflows/types.py
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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