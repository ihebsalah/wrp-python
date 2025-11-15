# wrp/shared/exceptions.py
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wrp.types import ErrorData


class WrpError(Exception):
    """
    Exception type raised when an error arrives over an WRP connection.
    """

    error: ErrorData

    def __init__(self, error: ErrorData):
        """Initialize WrpError."""
        super().__init__(error.message)
        self.error = error
