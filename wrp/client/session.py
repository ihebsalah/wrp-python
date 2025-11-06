# wrp/client/session.py
import logging
from datetime import timedelta
from typing import Any, Protocol

import anyio.lowlevel
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from jsonschema import SchemaError, ValidationError, validate
from pydantic import AnyUrl, TypeAdapter

from mcp.shared.message import ServerMessageMetadata, SessionMessage

import wrp.types as types
from wrp.shared.context import RequestContext
from wrp.shared.session import BaseSession, ProgressFnT, RequestResponder
from wrp.shared.version import LATEST_PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS

DEFAULT_CLIENT_INFO = types.Implementation(name="wrp", version="0.1.0")

logger = logging.getLogger("client")


class ElicitationFnT(Protocol):
    async def __call__(
        self,
        context: RequestContext["ClientSession", Any],
        params: types.ElicitRequestParams,
    ) -> types.ElicitResult | types.ErrorData: ...


class ListRootsFnT(Protocol):
    async def __call__(
        self, context: RequestContext["ClientSession", Any]
    ) -> types.ListRootsResult | types.ErrorData: ...


class LoggingFnT(Protocol):
    async def __call__(
        self,
        params: types.LoggingMessageNotificationParams,
    ) -> None: ...


class MessageHandlerFnT(Protocol):
    async def __call__(
        self,
        message: RequestResponder[types.ServerRequest, types.ClientResult] | types.ServerNotification | Exception,
    ) -> None: ...


async def _default_message_handler(
    message: RequestResponder[types.ServerRequest, types.ClientResult] | types.ServerNotification | Exception,
) -> None:
    await anyio.lowlevel.checkpoint()


async def _default_elicitation_callback(
    context: RequestContext["ClientSession", Any],
    params: types.ElicitRequestParams,
) -> types.ElicitResult | types.ErrorData:
    return types.ErrorData(
        code=types.INVALID_REQUEST,
        message="Elicitation not supported",
    )


async def _default_list_roots_callback(
    context: RequestContext["ClientSession", Any],
) -> types.ListRootsResult | types.ErrorData:
    return types.ErrorData(
        code=types.INVALID_REQUEST,
        message="List roots not supported",
    )


async def _default_logging_callback(
    params: types.LoggingMessageNotificationParams,
) -> None:
    pass


ClientResponse: TypeAdapter[types.ClientResult | types.ErrorData] = TypeAdapter(
    types.ClientResult | types.ErrorData
)


class ClientSession(
    BaseSession[
        types.ClientRequest,
        types.ClientNotification,
        types.ClientResult,
        types.ServerRequest,
        types.ServerNotification,
    ]
):
    def __init__(
        self,
        read_stream: MemoryObjectReceiveStream[SessionMessage | Exception],
        write_stream: MemoryObjectSendStream[SessionMessage],
        read_timeout_seconds: timedelta | None = None,
        elicitation_callback: ElicitationFnT | None = None,
        list_roots_callback: ListRootsFnT | None = None,
        logging_callback: LoggingFnT | None = None,
        message_handler: MessageHandlerFnT | None = None,
        client_info: types.Implementation | None = None,
    ) -> None:
        super().__init__(
            read_stream,
            write_stream,
            types.ServerRequest,
            types.ServerNotification,
            read_timeout_seconds=read_timeout_seconds,
        )
        self._client_info = client_info or DEFAULT_CLIENT_INFO
        self._elicitation_callback = elicitation_callback or _default_elicitation_callback
        self._list_roots_callback = list_roots_callback or _default_list_roots_callback
        self._logging_callback = logging_callback or _default_logging_callback
        self._message_handler = message_handler or _default_message_handler
        self._workflow_input_schemas: dict[str, dict[str, Any]] = {}
        self._workflow_output_schemas: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> types.InitializeResult:
        elicitation = (
            types.ElicitationCapability()
            if self._elicitation_callback is not _default_elicitation_callback
            else None
        )
        roots = (
            # TODO: Should this be based on whether we
            # _will_ send notifications, or only whether
            # they're supported?
            types.RootsCapability(listChanged=True)
            if self._list_roots_callback is not _default_list_roots_callback
            else None
        )

        result = await self.send_request(
            types.ClientRequest(
                types.InitializeRequest(
                    params=types.InitializeRequestParams(
                        protocolVersion=LATEST_PROTOCOL_VERSION,
                        capabilities=types.ClientCapabilities(
                            elicitation=elicitation,
                            experimental=None,
                            roots=roots,
                        ),
                        clientInfo=self._client_info,
                    ),
                )
            ),
            types.InitializeResult,
        )

        if result.protocolVersion not in SUPPORTED_PROTOCOL_VERSIONS:
            raise RuntimeError(f"Unsupported protocol version from the server: {result.protocolVersion}")

        await self.send_notification(types.ClientNotification(types.InitializedNotification()))

        return result

    async def send_ping(self) -> types.EmptyResult:
        """Send a ping request."""
        return await self.send_request(
            types.ClientRequest(types.PingRequest()),
            types.EmptyResult,
        )

    async def send_progress_notification(
        self,
        progress_token: str | int,
        progress: float,
        total: float | None = None,
        message: str | None = None,
    ) -> None:
        """Send a progress notification."""
        await self.send_notification(
            types.ClientNotification(
                types.ProgressNotification(
                    params=types.ProgressNotificationParams(
                        progressToken=progress_token,
                        progress=progress,
                        total=total,
                        message=message,
                    ),
                ),
            )
        )

    async def set_logging_level(self, level: types.LoggingLevel) -> types.EmptyResult:
        """Send a logging/setLevel request."""
        return await self.send_request(
            types.ClientRequest(
                types.SetLevelRequest(
                    params=types.SetLevelRequestParams(level=level),
                )
            ),
            types.EmptyResult,
        )

    async def list_resources(self, cursor: str | None = None) -> types.ListResourcesResult:
        """Send a resources/list request."""
        return await self.send_request(
            types.ClientRequest(
                types.ListResourcesRequest(
                    params=types.PaginatedRequestParams(cursor=cursor) if cursor is not None else None,
                )
            ),
            types.ListResourcesResult,
        )

    async def list_resource_templates(self, cursor: str | None = None) -> types.ListResourceTemplatesResult:
        """Send a resources/templates/list request."""
        return await self.send_request(
            types.ClientRequest(
                types.ListResourceTemplatesRequest(
                    params=types.PaginatedRequestParams(cursor=cursor) if cursor is not None else None,
                )
            ),
            types.ListResourceTemplatesResult,
        )

    async def read_resource(self, uri: AnyUrl) -> types.ReadResourceResult:
        """Send a resources/read request."""
        return await self.send_request(
            types.ClientRequest(
                types.ReadResourceRequest(
                    params=types.ReadResourceRequestParams(uri=uri),
                )
            ),
            types.ReadResourceResult,
        )

    async def subscribe_resource(self, uri: AnyUrl) -> types.EmptyResult:
        """Send a resources/subscribe request."""
        return await self.send_request(
            types.ClientRequest(
                types.SubscribeRequest(
                    params=types.SubscribeRequestParams(uri=uri),
                )
            ),
            types.EmptyResult,
        )

    async def unsubscribe_resource(self, uri: AnyUrl) -> types.EmptyResult:
        """Send a resources/unsubscribe request."""
        return await self.send_request(
            types.ClientRequest(
                types.UnsubscribeRequest(
                    params=types.UnsubscribeRequestParams(uri=uri),
                )
            ),
            types.EmptyResult,
        )

    async def list_workflows(self) -> types.ListWorkflowsResult:
        """Send a workflows/list request (WRP)."""
        result = await self.send_request(
            types.ClientRequest(types.ListWorkflowsRequest()),
            types.ListWorkflowsResult,
        )
        # refresh caches
        self._workflow_input_schemas.clear()
        self._workflow_output_schemas.clear()
        for wf in result.workflows:
            self._workflow_input_schemas[wf.name] = wf.inputSchema
            self._workflow_output_schemas[wf.name] = wf.outputSchema
        return result

    async def run_workflow(
        self,
        name: str,
        wf_input: dict[str, Any] | None = None,
        read_timeout_seconds: timedelta | None = None,
        progress_callback: ProgressFnT | None = None,
        validate_io: bool = True,
        *,
        # New: WRP run options (any of these may be provided)
        wrp_thread: str | None = None,
        wrp_conversation_seeding: dict[str, Any] | str | None = None,  # e.g. {"kind":"window","messages":20} or "none"
        wrp_run_filter: dict[str, Any] | None = None,            # e.g. {"since_run_id":"005", ...}
        wrp_request_context: dict[str, Any] | None = None,       # full override for request_context if you prefer
    ) -> types.RunWorkflowResult:
        """Send a workflows/run request (WRP) with optional client-side validation."""
        if validate_io:
            # Ensure we have schemas
            if name not in self._workflow_input_schemas:
                await self.list_workflows()
            schema = self._workflow_input_schemas.get(name)
            if schema:
                try:
                    validate(wf_input or {}, schema)
                except ValidationError as e:
                    raise RuntimeError(f"Invalid workflow input for {name}: {e}")
                except SchemaError as e:
                    raise RuntimeError(f"Invalid workflow input schema for {name}: {e}")

        # Build request_context for the server to parse (if any run option was provided
        # or a full override dict is supplied).
        request_context: dict[str, Any] | None = wrp_request_context
        if request_context is None and any([wrp_thread, wrp_conversation_seeding, wrp_run_filter]):
            wrp_ns: dict[str, Any] = {}
            if wrp_thread is not None:
                wrp_ns["thread"] = wrp_thread
            if wrp_conversation_seeding is not None:
                wrp_ns["conversation_seeding"] = wrp_conversation_seeding
            if wrp_run_filter is not None:
                wrp_ns["run_filter"] = wrp_run_filter
            request_context = {"wrp": wrp_ns}

        metadata = ServerMessageMetadata(request_context=request_context) if request_context else None

        result = await self.send_request(
            types.ClientRequest(
                types.RunWorkflowRequest(
                    params=types.RunWorkflowRequestParams(name=name, input=wf_input or {}),
                )
            ),
            types.RunWorkflowResult,
            request_read_timeout_seconds=read_timeout_seconds,
            progress_callback=progress_callback,
            metadata=metadata,  # <-- NEW: carry run options to the server
        )

        if validate_io and not result.isError:
            out_schema = self._workflow_output_schemas.get(name)
            if out_schema and result.output is not None:
                try:
                    validate(result.output, out_schema)
                except ValidationError as e:
                    raise RuntimeError(f"Invalid workflow output for {name}: {e}")
                except SchemaError as e:
                    raise RuntimeError(f"Invalid workflow output schema for {name}: {e}")
        return result

    async def send_roots_list_changed(self) -> None:
        """Send a roots/list_changed notification."""
        await self.send_notification(types.ClientNotification(types.RootsListChangedNotification()))

    async def _received_request(self, responder: RequestResponder[types.ServerRequest, types.ClientResult]) -> None:
        ctx = RequestContext[ClientSession, Any](
            request_id=responder.request_id,
            meta=responder.request_meta,
            session=self,
            lifespan_context=None,
        )

        match responder.request.root:
            case types.ElicitRequest(params=params):
                with responder:
                    response = await self._elicitation_callback(ctx, params)
                    client_response = ClientResponse.validate_python(response)
                    await responder.respond(client_response)

            case types.ListRootsRequest():
                with responder:
                    response = await self._list_roots_callback(ctx)
                    client_response = ClientResponse.validate_python(response)
                    await responder.respond(client_response)

            case types.PingRequest():
                with responder:
                    return await responder.respond(types.ClientResult(root=types.EmptyResult()))

    async def _handle_incoming(
        self,
        req: RequestResponder[types.ServerRequest, types.ClientResult] | types.ServerNotification | Exception,
    ) -> None:
        """Handle incoming messages by forwarding to the message handler."""
        await self._message_handler(req)

    async def _received_notification(self, notification: types.ServerNotification) -> None:
        """Handle notifications from the server."""
        # Process specific notification types
        match notification.root:
            case types.LoggingMessageNotification(params=params):
                await self._logging_callback(params)
            case types.WorkflowsListChangedNotification():
                # Invalidate caches so next call refreshes
                self._workflow_input_schemas.clear()
                self._workflow_output_schemas.clear()
            case _:
                pass