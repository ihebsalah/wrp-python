# wrp/shared/context.py
from dataclasses import dataclass
from typing import Any, Generic

from typing_extensions import TypeVar

from wrp.types import RequestId, RequestParams
from wrp.shared.session import BaseSession

SessionT = TypeVar("SessionT", bound=BaseSession[Any, Any, Any, Any, Any])
LifespanContextT = TypeVar("LifespanContextT")
RequestT = TypeVar("RequestT", default=Any)


@dataclass
class RequestContext(Generic[SessionT, LifespanContextT, RequestT]):
    request_id: RequestId
    meta: RequestParams.Meta | None
    session: SessionT
    lifespan_context: LifespanContextT
    request: RequestT | None = None
