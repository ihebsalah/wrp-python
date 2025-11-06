# wrp/server/runtime/exceptions.py
"""Custom exceptions for WRP."""

from wrp.shared.exceptions import WrpError

class InvalidSignature(WrpError):
    """Invalid signature"""

class WorkflowError(WrpError):
    """Generic workflow error."""

class RunStateError(RuntimeError):
    """Raised when mutating a run that has already concluded."""
    pass

class WorkflowMarkedFailure(RuntimeError):
    """Internal control-flow signal: author concluded run as 'failed' (business failure)."""
    pass

class WorkflowMarkedError(RuntimeError):
    """Internal control-flow signal: author concluded run as 'error' (technical fault)."""
    pass