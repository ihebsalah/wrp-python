# wrp/server/runtime/exceptions.py
"""Custom exceptions for WRP."""

from typing import Any
from wrp.shared.exceptions import WrpError
from wrp.types import ErrorData, INTERNAL_ERROR, INVALID_REQUEST

class InvalidSignature(WrpError):
    """Invalid signature"""

    def __init__(
        self,
        message: str,
        code: int | None = None,
        data: Any | None = None,
    ):
        super().__init__(
            ErrorData(
                code=code if code is not None else INVALID_REQUEST,
                message=message,
                data=data,
            )
        )


class WorkflowError(WrpError):
    """Generic workflow error."""

    def __init__(
        self,
        message: str,
        code: int | None = None,
        data: Any | None = None,
    ):
        super().__init__(
            ErrorData(
                code=code if code is not None else INTERNAL_ERROR,
                message=message,
                data=data,
            )
        )

class RunStateError(RuntimeError):
    """Raised when mutating a run that has already concluded."""
    pass

class WorkflowMarkedFailure(RuntimeError):
    """Internal control-flow signal: author concluded run as 'failed' (business failure)."""
    pass

class WorkflowMarkedError(RuntimeError):
    """Internal control-flow signal: author concluded run as 'error' (technical fault)."""
    pass