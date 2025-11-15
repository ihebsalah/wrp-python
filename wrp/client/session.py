# wrp/client/session.py
import logging
from datetime import timedelta
from typing import Any, Optional, Protocol

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

    # -------- System Events: client helpers ----------
    def _build_events_selectors(
        self,
        *,
        system_session_id: str | None,
        run_id: str | None,
        span_id: str | None,
        channel: str | None,
    ) -> tuple[
        types.RunsScope | None,
        types.SpanScope | None,
        types.ChannelScope | None,
        types.SystemSessionScope | None,
    ]:
        """Build scope objects from IDs for system event subscriptions."""
        if span_id is not None:
            if not (system_session_id and run_id):
                raise ValueError("span subscription requires system_session_id and run_id")
            return (
                None,
                types.SpanScope(system_session_id=system_session_id, run_id=run_id, span_id=span_id),
                None,
                None,
            )
        if channel is not None:
            if not (system_session_id and run_id):
                raise ValueError("channel subscription requires system_session_id and run_id")
            return (
                None,
                None,
                types.ChannelScope(
                    system_session_id=system_session_id, run_id=run_id, channel=channel
                ),
                None,
            )
        if run_id is not None:
            if not system_session_id:
                raise ValueError("runs subscription requires system_session_id")
            return (types.RunsScope(system_session_id=system_session_id, run_id=run_id), None, None, None)
        if system_session_id is not None:
            return (None, None, None, types.SystemSessionScope(system_session_id=system_session_id))
        return (None, None, None, None)

    async def system_events_subscribe(
        self,
        *,
        topic: types.Topic,
        system_session_id: str | None = None,
        run_id: str | None = None,
        span_id: str | None = None,
        channel: str | None = None,
        deliver_initial: bool | None = None,
        coalesce_ms: int | None = None,
    ) -> types.SystemEventsSubscribeResult:
        """Send a system/events/subscribe request with convenient selectors."""
        runs, span, ch, sess = self._build_events_selectors(
            system_session_id=system_session_id, run_id=run_id, span_id=span_id, channel=channel
        )
        options = None
        if deliver_initial is not None or coalesce_ms is not None:
            options = types.SubscribeOptions(deliverInitial=deliver_initial, coalesceMs=coalesce_ms)
        params = types.SystemEventsSubscribeParams(
            topic=topic, runs=runs, span=span, channel=ch, session=sess, options=options
        )
        return await self.send_request(
            types.ClientRequest(types.SystemEventsSubscribeRequest(params=params)),
            types.SystemEventsSubscribeResult,
        )

    async def system_events_unsubscribe(
        self,
        *,
        subscription_id: str | None = None,
        topic: types.Topic | None = None,
        system_session_id: str | None = None,
        run_id: str | None = None,
        span_id: str | None = None,
        channel: str | None = None,
    ) -> types.EmptyResult:
        """Send a system/events/unsubscribe request with convenient selectors."""
        runs, span, ch, sess = self._build_events_selectors(
            system_session_id=system_session_id, run_id=run_id, span_id=span_id, channel=channel
        )
        if not subscription_id and not topic and not any([runs, span, ch, sess]):
            raise ValueError("unsubscribe requires subscription_id or (topic + selector/global)")
        params = types.SystemEventsUnsubscribeParams(
            subscriptionId=subscription_id, topic=topic, runs=runs, span=span, channel=ch, session=sess
        )
        return await self.send_request(
            types.ClientRequest(types.SystemEventsUnsubscribeRequest(params=params)),
            types.EmptyResult,
        )

    # -------- System handlers: reads ----------
    async def runs_list(
        self,
        *,
        system_session_id: str,
        workflow: str | None = None,
        thread_id: str | None = None,
        state: types.RunState | None = None,
        outcome: types.RunOutcome | None = None,
    ) -> types.RunsListResult:
        """Send a runs/list request with optional filters."""
        params = types.RunsListRequestParams(
            system_session=types.SystemSessionScope(system_session_id=system_session_id),
            workflow=workflow,
            thread_id=thread_id,
            state=state,
            outcome=outcome,
        )
        return await self.send_request(
            types.ClientRequest(types.RunsListRequest(params=params)),
            types.RunsListResult,
        )

    async def runs_read(self, *, system_session_id: str, run_id: str) -> types.RunsReadResult:
        """Send a runs/read request for a specific run."""
        runs = types.RunsScope(system_session_id=system_session_id, run_id=run_id)
        params = types.RunsReadRequestParams(runs=runs)
        return await self.send_request(
            types.ClientRequest(types.RunsReadRequest(params=params)),
            types.RunsReadResult,
        )

    async def runs_input_read(self, *, system_session_id: str, run_id: str) -> types.RunsIOReadResult:
        """Send a runs/input/read request."""
        runs = types.RunsScope(system_session_id=system_session_id, run_id=run_id)
        return await self.send_request(
            types.ClientRequest(types.RunsInputReadRequest(params=types.RunsIOReadRequestParams(runs=runs))),
            types.RunsIOReadResult,
        )

    async def runs_output_read(self, *, system_session_id: str, run_id: str) -> types.RunsIOReadResult:
        """Send a runs/output/read request."""
        runs = types.RunsScope(system_session_id=system_session_id, run_id=run_id)
        return await self.send_request(
            types.ClientRequest(types.RunsOutputReadRequest(params=types.RunsIOReadRequestParams(runs=runs))),
            types.RunsIOReadResult,
        )

    async def telemetry_spans_list(self, *, system_session_id: str, run_id: str) -> types.TelemetrySpansListResult:
        """Send a telemetry/spans/list request."""
        runs = types.RunsScope(system_session_id=system_session_id, run_id=run_id)
        return await self.send_request(
            types.ClientRequest(
                types.TelemetrySpansListRequest(params=types.TelemetrySpansListRequestParams(runs=runs))
            ),
            types.TelemetrySpansListResult,
        )

    async def telemetry_span_read(
        self, *, system_session_id: str, run_id: str, span_id: str
    ) -> types.TelemetrySpanReadResult:
        """Send a telemetry/span/read request."""
        span = types.SpanScope(system_session_id=system_session_id, run_id=run_id, span_id=span_id)
        return await self.send_request(
            types.ClientRequest(types.TelemetrySpanReadRequest(params=types.TelemetrySpanReadRequestParams(span=span))),
            types.TelemetrySpanReadResult,
        )

    async def telemetry_payload_read(
        self, *, system_session_id: str, run_id: str, span_id: str
    ) -> types.TelemetryPayloadReadResult:
        """Send a telemetry/payload/read request."""
        span = types.SpanScope(system_session_id=system_session_id, run_id=run_id, span_id=span_id)
        return await self.send_request(
            types.ClientRequest(
                types.TelemetryPayloadReadRequest(params=types.TelemetryPayloadReadRequestParams(span=span))
            ),
            types.TelemetryPayloadReadResult,
        )

    async def conversations_channels_list(
        self,
        *,
        system_session_id: str,
        run_id: str,
    ) -> types.ChannelsListResult:
        """Send a conversations/channels/list request."""
        runs = types.RunsScope(system_session_id=system_session_id, run_id=run_id)
        return await self.send_request(
            types.ClientRequest(
                types.ChannelsListRequest(
                    params=types.ChannelsListRequestParams(runs=runs)
                )
            ),
            types.ChannelsListResult,
        )

    async def conversations_channel_read(
        self,
        *,
        system_session_id: str,
        run_id: str,
        channel: str,
    ) -> types.ChannelReadResult:
        """Send a conversations/channel/read request."""
        scope = types.ChannelScope(
            system_session_id=system_session_id,
            run_id=run_id,
            channel=channel,
        )
        return await self.send_request(
            types.ClientRequest(
                types.ChannelReadRequest(
                    params=types.ChannelReadRequestParams(channel=scope)
                )
            ),
            types.ChannelReadResult,
        )

    async def system_sessions_list(self) -> types.SystemSessionsListResult:
        """Send a system/sessions/list request."""
        return await self.send_request(
            types.ClientRequest(types.SystemSessionsListRequest()),
            types.SystemSessionsListResult,
        )

    async def system_session_read(self, *, system_session_id: str) -> types.SystemSessionReadResult:
        """Send a system/session/read request."""
        scope = types.SystemSessionScope(system_session_id=system_session_id)
        return await self.send_request(
            types.ClientRequest(
                types.SystemSessionReadRequest(params=types.SystemSessionReadRequestParams(session=scope))
            ),
            types.SystemSessionReadResult,
        )

    # -------- Provider settings (WRP) ----------
    async def provider_settings_read(self, provider: str) -> types.ProviderSettingsReadResult:
        """Read effective settings for a provider."""
        params = types.ProviderSettingsReadRequestParams(provider=provider)
        return await self.send_request(
            types.ClientRequest(types.ProviderSettingsReadRequest(params=params)),
            types.ProviderSettingsReadResult,
        )

    async def provider_settings_schema(self, provider: str) -> types.ProviderSettingsSchemaResult:
        """Fetch JSON schema for a provider's settings model."""
        params = types.ProviderSettingsSchemaRequestParams(provider=provider)
        return await self.send_request(
            types.ClientRequest(types.ProviderSettingsSchemaRequest(params=params)),
            types.ProviderSettingsSchemaResult,
        )

    async def provider_settings_update(
        self,
        provider: str,
        values: dict[str, Any],
    ) -> types.ProviderSettingsReadResult:
        """Merge/update provider settings."""
        params = types.ProviderSettingsUpdateRequestParams(provider=provider, values=values)
        return await self.send_request(
            types.ClientRequest(types.ProviderSettingsUpdateRequest(params=params)),
            types.ProviderSettingsReadResult,
        )

    # -------- Agent settings (WRP) ----------
    async def agent_settings_read(self, agent: str) -> types.AgentSettingsReadResult:
        """Read effective settings for an agent."""
        params = types.AgentSettingsReadRequestParams(agent=agent)
        return await self.send_request(
            types.ClientRequest(types.AgentSettingsReadRequest(params=params)),
            types.AgentSettingsReadResult,
        )

    async def agent_settings_schema(self, agent: str) -> types.AgentSettingsSchemaResult:
        """Fetch JSON schema for an agent's settings model."""
        params = types.AgentSettingsSchemaRequestParams(agent=agent)
        return await self.send_request(
            types.ClientRequest(types.AgentSettingsSchemaRequest(params=params)),
            types.AgentSettingsSchemaResult,
        )

    async def agent_settings_update(
        self,
        agent: str,
        values: dict[str, Any],
    ) -> types.AgentSettingsReadResult:
        """Merge/update agent settings."""
        params = types.AgentSettingsUpdateRequestParams(agent=agent, values=values)
        return await self.send_request(
            types.ClientRequest(types.AgentSettingsUpdateRequest(params=params)),
            types.AgentSettingsReadResult,
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
        wrp_seeding_run_filter: dict[str, Any] | None = None,  # e.g. {"since_run_id":"005", ...}
        wrp_request_context: dict[str, Any] | None = None,  # full override for request_context if you prefer
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
        if request_context is None and any([wrp_thread, wrp_conversation_seeding, wrp_seeding_run_filter]):
            wrp_ns: dict[str, Any] = {}
            if wrp_thread is not None:
                wrp_ns["thread"] = wrp_thread
            if wrp_conversation_seeding is not None:
                wrp_ns["conversation_seeding"] = wrp_conversation_seeding
            if wrp_seeding_run_filter is not None:
                wrp_ns["seeding_run_filter"] = wrp_seeding_run_filter
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
            case types.SystemEventsUpdatedNotification():
                # Handled by message_handler via _handle_incoming; no-op here
                pass
            case _:
                pass