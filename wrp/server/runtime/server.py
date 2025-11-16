# wrp/server/runtime/server.py
from __future__ import annotations as _annotations

import inspect
import re
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Iterable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Any, Generic, Literal, cast

import anyio
from pydantic import BaseModel
from pydantic.networks import AnyUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send

from mcp.server.auth.middleware.auth_context import AuthContextMiddleware
from mcp.server.auth.middleware.bearer_auth import BearerAuthBackend, RequireAuthMiddleware
from mcp.server.auth.provider import OAuthAuthorizationServerProvider, ProviderTokenVerifier, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp.exceptions import ResourceError
from mcp.server.fastmcp.utilities.context_injection import find_context_parameter
from mcp.server.fastmcp.utilities.logging import configure_logging, get_logger
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.streamable_http import EventStore
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import AnyFunction
from mcp.types import DEFAULT_NEGOTIATED_VERSION as MCP_DEFAULT_PROTOCOL

from wrp.server.sse import SseServerTransport
from wrp.server.stdio import stdio_server
from wrp.server.elicitation import ElicitationResult, ElicitSchemaModelT, elicit_with_validation
from wrp.server.lowlevel.server import LifespanResultT
from wrp.server.lowlevel.server import Server as WRPServer
from wrp.server.lowlevel.server import lifespan as default_lifespan
from wrp.server.middleware.header_compat import HeaderCompatMiddleware
from wrp.server.runtime.resources import FunctionResource, Resource, ResourceManager
from wrp.server.runtime.settings.agents import AgentSettingsRegistry
from wrp.server.runtime.settings.agents import AgentSettings
from wrp.server.runtime.conversations.privacy.guards import (
    is_private_only_conversations_selector,
)
from wrp.server.runtime.conversations.privacy.policy import ConversationResourcePolicy
from wrp.server.runtime.conversations.privacy.redaction import sanitize_conversation_items
from wrp.server.runtime.conversations.seeding import WorkflowConversationSeeding
from wrp.server.runtime.conversations.types import ChannelMeta, ChannelView
from wrp.server.runtime.limits import DEFAULT_GLOBAL_INPUT_LIMIT_BYTES
from wrp.server.runtime.settings.providers import ProviderSettingsRegistry
from wrp.server.runtime.settings.providers import ProviderSettings
from wrp.server.runtime.settings.bootstrap import hydrate_provider_and_agent_settings
from wrp.server.runtime.runs.bindings import RunBindings
from wrp.server.runtime.store.base import Store
from wrp.server.runtime.store.stores.memory_store import InMemoryStore
from wrp.server.runtime.telemetry.payloads.types import SpanPayloadEnvelope
from wrp.server.runtime.telemetry.privacy.guards import is_private_only_span_payload_uri
from wrp.server.runtime.telemetry.privacy.policy import TelemetryResourcePolicy
from wrp.server.runtime.telemetry.privacy.redaction import sanitize_envelope_dict
from wrp.server.runtime.telemetry.views import build_span_index, get_span_view, list_span_views
from wrp.server.runtime.workflows import WorkflowManager
from wrp.server.runtime.settings.workflows import WorkflowSettings
from wrp.server.runtime.workflows.types import RunWorkflowResult, WorkflowInput, WorkflowOutput
from wrp.server.session import ServerSession, ServerSessionT
from wrp.shared.context import LifespanContextT, RequestContext, RequestT
from wrp.shared.version import SUPPORTED_PROTOCOL_VERSIONS as WRP_SUPPORTED_VERSIONS
import wrp.types as types

if TYPE_CHECKING:
    from mcp.server.lowlevel.server import Server as MCPServer

logger = get_logger(__name__)


class Settings(BaseSettings, Generic[LifespanResultT]):
    """WRP server settings.

    All settings can be configured via environment variables with the prefix WRP_.
    For example, WRP_DEBUG=true will set debug=True.
    """

    model_config = SettingsConfigDict(
        env_prefix="WRP_",
        env_file=".env",
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,
        extra="ignore",
    )

    # Server settings
    debug: bool
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    # HTTP settings
    host: str
    port: int
    mount_path: str
    sse_path: str
    message_path: str
    streamable_http_path: str

    # StreamableHTTP settings
    json_response: bool
    stateless_http: bool
    """Define if the server should create a new transport per request."""

    # resource settings
    warn_on_duplicate_resources: bool

    # workflow settings
    warn_on_duplicate_workflows: bool

    # TODO(Marcelo): Investigate if this is used. If it is, it's probably a good idea to remove it.
    dependencies: list[str]
    """A list of dependencies to install in the server environment."""

    lifespan: Callable[[WRP[LifespanResultT]], AbstractAsyncContextManager[LifespanResultT]] | None
    """A async context manager that will be called when the server is started."""

    auth: AuthSettings | None

    # Transport security settings (DNS rebinding protection)
    transport_security: TransportSecuritySettings | None
    # Limits
    # Default safeguard for *all* servers (overridable via env or ctor).
    # Env var: WRP_GLOBAL_INPUT_LIMIT_BYTES
    global_input_limit_bytes: int | None = DEFAULT_GLOBAL_INPUT_LIMIT_BYTES


def lifespan_wrapper(
    app: WRP[LifespanResultT],
    lifespan: Callable[[WRP[LifespanResultT]], AbstractAsyncContextManager[LifespanResultT]],
) -> Callable[[WRPServer[LifespanResultT, Request]], AbstractAsyncContextManager[LifespanResultT]]:
    @asynccontextmanager
    async def wrap(_: WRPServer[LifespanResultT, Request]) -> AsyncIterator[LifespanResultT]:
        async with lifespan(app) as context:
            yield context

    return wrap


class WRP(Generic[LifespanResultT]):
    def __init__(  # noqa: PLR0913
        self,
        name: str | None = None,
        instructions: str | None = None,
        website_url: str | None = None,
        icons: list[types.Icon] | None = None,
        auth_server_provider: OAuthAuthorizationServerProvider[Any, Any, Any] | None = None,
        token_verifier: TokenVerifier | None = None,
        event_store: EventStore | None = None,
        *,
        debug: bool = False,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
        host: str = "127.0.0.1",
        port: int = 8000,
        mount_path: str = "/",
        sse_path: str = "/sse",
        message_path: str = "/messages/",
        streamable_http_path: str = "/wrp",
        json_response: bool = False,
        stateless_http: bool = False,
        warn_on_duplicate_resources: bool = True,
        warn_on_duplicate_workflows: bool = True,
        dependencies: Collection[str] = (),
        lifespan: Callable[[WRP[LifespanResultT]], AbstractAsyncContextManager[LifespanResultT]] | None = None,
        auth: AuthSettings | None = None,
        transport_security: TransportSecuritySettings | None = None,
        store: Store | None = None,
        telemetry_policy: TelemetryResourcePolicy | None = None,
        conversation_policy: ConversationResourcePolicy | None = None,
        default_seeding: WorkflowConversationSeeding | None = None,
        # None => use Settings default or env override
        global_input_limit_bytes: int | None = None,
    ):
        # Build Settings while *only* overriding the default if a value was provided.
        _settings_kwargs = dict[str, Any](
            debug=debug,
            log_level=log_level,
            host=host,
            port=port,
            mount_path=mount_path,
            sse_path=sse_path,
            message_path=message_path,
            streamable_http_path=streamable_http_path,
            json_response=json_response,
            stateless_http=stateless_http,
            warn_on_duplicate_resources=warn_on_duplicate_resources,
            warn_on_duplicate_workflows=warn_on_duplicate_workflows,
            dependencies=list(dependencies),
            lifespan=lifespan,
            auth=auth,
            transport_security=transport_security,
        )
        if global_input_limit_bytes is not None:
            _settings_kwargs["global_input_limit_bytes"] = global_input_limit_bytes
        self.settings = Settings(**_settings_kwargs)

        self._wrp_server = WRPServer(
            name=name or "WRP",
            instructions=instructions,
            website_url=website_url,
            icons=icons,
            # TODO(Marcelo): It seems there's a type mismatch between the lifespan type from a WRP and Server.
            # We need to create a Lifespan type that is a generic on the server type, like Starlette does.
            lifespan=(lifespan_wrapper(self, self.settings.lifespan) if self.settings.lifespan else default_lifespan),  # type: ignore
        )
        self._resource_manager = ResourceManager(warn_on_duplicate_resources=self.settings.warn_on_duplicate_resources)
        # run store (pluggable) - create before manager so it can be injected
        self._store: Store = store or InMemoryStore()
        # workflow manager with store for settings persistence
        self._workflow_manager = WorkflowManager(
            warn_on_duplicate_workflows=self.settings.warn_on_duplicate_workflows,
            store=self._store,
        )
        # provider/agent settings registries (global, non-run)
        self._provider_settings_registry = ProviderSettingsRegistry(store=self._store)
        self._agent_settings_registry = AgentSettingsRegistry(store=self._store)
        # Validate auth configuration
        if self.settings.auth is not None:
            if auth_server_provider and token_verifier:
                raise ValueError("Cannot specify both auth_server_provider and token_verifier")
            if not auth_server_provider and not token_verifier:
                raise ValueError("Must specify either auth_server_provider or token_verifier when auth is enabled")
        elif auth_server_provider or token_verifier:
            raise ValueError("Cannot specify auth_server_provider or token_verifier without auth settings")

        self._auth_server_provider = auth_server_provider
        self._token_verifier = token_verifier

        # Create token verifier from provider if needed (backwards compatibility)
        if auth_server_provider and not token_verifier:
            self._token_verifier = ProviderTokenVerifier(auth_server_provider)
        self._event_store = event_store
        self._custom_starlette_routes: list[Route] = []
        self.dependencies = self.settings.dependencies
        self._session_manager: StreamableHTTPSessionManager | None = None
        # Telemetry serving policy
        self._telemetry_policy: TelemetryResourcePolicy = telemetry_policy or TelemetryResourcePolicy.defaults()
        self._conversation_policy: ConversationResourcePolicy = (
            conversation_policy or ConversationResourcePolicy.defaults()
        )
        # Global seeding fallback for all workflows (can be overridden per workflow)
        self._default_seeding: WorkflowConversationSeeding | None = default_seeding
        # Resource subscriptions (authors' resources only)
        self._resource_subscriptions: dict[str, "weakref.WeakSet[ServerSession]"] = {}
        # System events subscriptions: key -> {seq:int, subs:WeakSet}
        self._event_subscriptions: dict[str, dict[str, Any]] = {}
        # subscriptionId -> key
        self._event_subscription_index: dict[str, str] = {}

        # Set up WRP protocol handlers
        self._setup_handlers()

        # Configure logging
        configure_logging(self.settings.log_level)

    @property
    def name(self) -> str:
        return self._wrp_server.name

    @property
    def instructions(self) -> str | None:
        return self._wrp_server.instructions

    @property
    def website_url(self) -> str | None:
        return self._wrp_server.website_url

    @property
    def icons(self) -> list[types.Icon] | None:
        return self._wrp_server.icons

    @property
    def session_manager(self) -> StreamableHTTPSessionManager:
        """Get the StreamableHTTP session manager.

        This is exposed to enable advanced use cases like mounting multiple
        WRP servers in a single FastAPI application.

        Raises:
            RuntimeError: If called before streamable_http_app() has been called.
        """
        if self._session_manager is None:
            raise RuntimeError(
                "Session manager can only be accessed after"
                "calling streamable_http_app()."
                "The session manager is created lazily"
                "to avoid unnecessary initialization."
            )
        return self._session_manager

    def run(
        self,
        transport: Literal["stdio", "sse", "streamable-http"] = "stdio",
        mount_path: str | None = None,
    ) -> None:
        """Run the WRP server. Note this is a synchronous function.

        Args:
            transport: Transport protocol to use ("stdio", "sse", or "streamable-http")
            mount_path: Optional mount path for SSE transport
        """
        TRANSPORTS = Literal["stdio", "sse", "streamable-http"]
        if transport not in TRANSPORTS.__args__:  # type: ignore
            raise ValueError(f"Unknown transport: {transport}")

        match transport:
            case "stdio":
                anyio.run(self.run_stdio_async)
            case "sse":
                anyio.run(lambda: self.run_sse_async(mount_path))
            case "streamable-http":
                anyio.run(self.run_streamable_http_async)

    def _setup_handlers(self) -> None:
        """Set up core WRP protocol handlers."""
        self._wrp_server.list_resources()(self._list_resources)
        self._wrp_server.read_resource()(self._read_resource)
        self._wrp_server.list_resource_templates()(self._list_resource_templates)
        # subscriptions
        self._wrp_server.subscribe_resource()(self._subscribe_resource)
        self._wrp_server.unsubscribe_resource()(self._unsubscribe_resource)

        self._wrp_server.list_workflows()(self._list_workflows)
        self._wrp_server.run_workflow(validate_input=True)(self._run_workflow)
        # workflow settings (lowlevel protocol handlers)
        self._wrp_server.workflow_settings_read()(self._wf_settings_read)
        self._wrp_server.workflow_settings_schema()(self._wf_settings_schema)
        self._wrp_server.workflow_settings_update()(self._wf_settings_update)
        # provider settings (lowlevel protocol handlers)
        self._wrp_server.provider_settings_read()(self._provider_settings_read)
        self._wrp_server.provider_settings_schema()(self._provider_settings_schema)
        self._wrp_server.provider_settings_update()(self._provider_settings_update)
        # agent settings (lowlevel protocol handlers)
        self._wrp_server.agent_settings_read()(self._agent_settings_read)
        self._wrp_server.agent_settings_schema()(self._agent_settings_schema)
        self._wrp_server.agent_settings_update()(self._agent_settings_update)
        # system events + handlers
        self._wrp_server.system_events_subscribe()(self._system_events_subscribe)
        self._wrp_server.system_events_unsubscribe()(self._system_events_unsubscribe)
        self._wrp_server.runs_list()(self._runs_list)
        self._wrp_server.runs_read()(self._runs_read)
        self._wrp_server.runs_input_read()(self._runs_input_read)
        self._wrp_server.runs_output_read()(self._runs_output_read)
        self._wrp_server.telemetry_spans_list()(self._telemetry_spans_list)
        self._wrp_server.telemetry_span_read()(self._telemetry_span_read)
        self._wrp_server.telemetry_payload_read()(self._telemetry_payload_read)
        self._wrp_server.conversations_channels_list()(self._conversations_channels_list)
        self._wrp_server.conversations_channel_read()(self._conversations_channel_read)
        self._wrp_server.system_sessions_list()(self._system_sessions_list)
        self._wrp_server.system_session_read()(self._system_session_read)

    async def _subscribe_resource(self, uri: AnyUrl) -> None:
        """Remember that the *current* session wants updates for this author-provided resource."""
        ctx = self.get_context()
        sess = ctx.request_context.session
        self._resource_subscriptions.setdefault(str(uri), weakref.WeakSet()).add(sess)

    async def _unsubscribe_resource(self, uri: AnyUrl) -> None:
        """Remove the *current* session from the uri's subscription set."""
        ctx = self.get_context()
        sess = ctx.request_context.session
        bucket = self._resource_subscriptions.get(str(uri))
        if bucket:
            try:
                bucket.remove(sess)  # no-op if not present
            except KeyError:
                pass

    # -------------------------
    # System Events: runtime impl
    # -------------------------
    def _system_event_key(
        self,
        topic: types.Topic,
        *,
        runs: types.RunsScope | None = None,
        span: types.SpanScope | None = None,
        channel: types.ChannelScope | None = None,
        session_sel: types.SystemSessionScope | None = None,
    ) -> str:
        if span:  # most specific
            return f"{topic}|ss:{span.system_session_id}|run:{span.run_id}|span:{span.span_id}"
        if channel:
            return f"{topic}|ss:{channel.system_session_id}|run:{channel.run_id}|ch:{channel.channel}"
        if runs:
            return f"{topic}|ss:{runs.system_session_id}|run:{runs.run_id}"
        if session_sel:
            return f"{topic}|ss:{session_sel.system_session_id}"
        return f"{topic}|global"

    async def _system_events_subscribe(self, params: types.SystemEventsSubscribeParams) -> types.SystemEventsSubscribeResult:
        # Enforce policy for conversations subscriptions
        if params.topic == "conversations/channel":
            ch_id = params.channel.channel if params.channel else None
            if is_private_only_conversations_selector(self._conversation_policy, channel=ch_id):
                raise PermissionError("Subscription denied: channel is private under current conversation policy.")
        elif params.topic == "conversations/channels":
            # Aggregate guard: if the entire channels index would be private-only, deny
            if is_private_only_conversations_selector(self._conversation_policy, channel=None):
                raise PermissionError(
                    "Subscription denied: channels index is private under current conversation policy."
                )

        ctx = self.get_context()
        sess = ctx.request_context.session
        key = self._system_event_key(
            params.topic, runs=params.runs, span=params.span, channel=params.channel, session_sel=params.session
        )
        bucket = self._event_subscriptions.setdefault(key, {"seq": 0, "subs": weakref.WeakSet()})
        bucket["subs"].add(sess)
        # assign an id and index it
        sub_id = f"sub_{id(sess)}_{len(self._event_subscription_index)+1}"
        self._event_subscription_index[sub_id] = key
        # initial seed
        if params.options and params.options.deliverInitial:
            bucket["seq"] += 1
            await sess.send_system_events_updated(
                topic=params.topic,
                sequence=bucket["seq"],
                change="refetch",
                runs=params.runs,
                span=params.span,
                channel=params.channel,
                session_sel=params.session,
            )
        return types.SystemEventsSubscribeResult(subscriptionId=sub_id)

    async def _system_events_unsubscribe(self, params: types.SystemEventsUnsubscribeParams) -> None:
        key = None
        if params.subscriptionId and params.subscriptionId in self._event_subscription_index:
            key = self._event_subscription_index.pop(params.subscriptionId, None)
        if key is None:
            key = (
                self._system_event_key(
                    params.topic, runs=params.runs, span=params.span, channel=params.channel, session_sel=params.session
                )
                if params.topic
                else None
            )
        if key and key in self._event_subscriptions:
            # best-effort: remove current session from bucket
            try:
                ctx = self.get_context()
                sess = ctx.request_context.session
                self._event_subscriptions[key]["subs"].discard(sess)  # type: ignore[index]
            except Exception:
                pass

    async def _emit_system_event(
        self,
        *,
        topic: types.Topic,
        change: types.ChangeKind | None,
        runs: types.RunsScope | None = None,
        span: types.SpanScope | None = None,
        channel: types.ChannelScope | None = None,
        session_sel: types.SystemSessionScope | None = None,
    ) -> None:
        """Fan-out an events/updated to matching subscribers."""
        key = self._system_event_key(topic, runs=runs, span=span, channel=channel, session_sel=session_sel)
        bucket = self._event_subscriptions.get(key)
        if not bucket:
            return
        bucket["seq"] += 1
        seq = bucket["seq"]
        subs = list(bucket["subs"])  # copy
        for s in subs:
            try:
                await s.send_system_events_updated(
                    topic=topic,
                    sequence=seq,
                    change=change,
                    runs=runs,
                    span=span,
                    channel=channel,
                    session_sel=session_sel,
                )
            except Exception:
                try:
                    bucket["subs"].discard(s)
                except Exception:
                    pass

    # -------------------------
    # System handler impls
    # -------------------------
    async def _find_run_span_id(self, system_session_id: str, run_id: str) -> str | None:
        events = await self.store.load_telemetry(system_session_id, run_id)
        for ev in events:
            if (
                getattr(ev, "kind", None) == "span"
                and getattr(ev, "span_kind", None) == "run"
                and getattr(ev, "phase", None) == "start"
            ):
                return getattr(ev, "span_id", None)
        return None

    async def _runs_list(self, params: types.RunsListRequestParams) -> types.RunsListResult:
        sess = params.system_session.system_session_id
        runs = await self.store.list_runs(
            sess,
            workflow_name=params.workflow,
            thread_id=params.thread_id,
            state=params.state,
            outcome=params.outcome,
        )
        return types.RunsListResult(runs=runs)

    async def _runs_read(self, runs: types.RunsScope) -> types.RunsReadResult:
        meta = await self.store.get_run(runs.system_session_id, runs.run_id)
        return types.RunsReadResult(run=meta)

    async def _runs_input_read(self, runs: types.RunsScope) -> types.RunsIOReadResult:
        system_session_id = runs.system_session_id
        run_id = runs.run_id
        span_id = await self._find_run_span_id(system_session_id, run_id)
        meta = await self.store.get_run(system_session_id, run_id)
        workflow = meta.workflow_name if meta else None
        wf = self._workflow_manager.get(workflow) if workflow else None
        schema = wf.input_schema if wf else None
        if not span_id:
            return types.RunsIOReadResult(
                data=None,
                workflow=workflow,
                jsonSchema=schema,
                system_session_id=system_session_id,
                run_id=run_id,
            )
        env = await self.store.get_span_payload(system_session_id, run_id, span_id)
        if not env:
            return types.RunsIOReadResult(
                data=None,
                workflow=workflow,
                jsonSchema=schema,
                system_session_id=system_session_id,
                run_id=run_id,
            )
        env_dict = env.model_dump()
        sanitized = sanitize_envelope_dict(env_dict, self._telemetry_policy)
        part = sanitized.get("capture", {}).get("start")
        return types.RunsIOReadResult(
            data=part.get("data") if part else None,
            workflow=workflow,
            jsonSchema=schema,
            system_session_id=system_session_id,
            run_id=run_id,
        )

    async def _runs_output_read(self, runs: types.RunsScope) -> types.RunsIOReadResult:
        system_session_id = runs.system_session_id
        run_id = runs.run_id
        span_id = await self._find_run_span_id(system_session_id, run_id)
        meta = await self.store.get_run(system_session_id, run_id)
        workflow = meta.workflow_name if meta else None
        wf = self._workflow_manager.get(workflow) if workflow else None
        schema = wf.output_schema if wf else None
        if not span_id:
            return types.RunsIOReadResult(
                data=None,
                workflow=workflow,
                jsonSchema=schema,
                system_session_id=system_session_id,
                run_id=run_id,
            )
        env = await self.store.get_span_payload(system_session_id, run_id, span_id)
        if not env:
            return types.RunsIOReadResult(
                data=None,
                workflow=workflow,
                jsonSchema=schema,
                system_session_id=system_session_id,
                run_id=run_id,
            )
        env_dict = env.model_dump()
        sanitized = sanitize_envelope_dict(env_dict, self._telemetry_policy)
        part = sanitized.get("capture", {}).get("end")
        return types.RunsIOReadResult(
            data=part.get("data") if part else None,
            workflow=workflow,
            jsonSchema=schema,
            system_session_id=system_session_id,
            run_id=run_id,
        )

    async def _telemetry_spans_list(self, runs: types.RunsScope) -> types.TelemetrySpansListResult:
        events = await self.store.load_telemetry(runs.system_session_id, runs.run_id, kinds={"span"})
        spans = build_span_index(
            events,
            mask_model=self._telemetry_policy.mask_model_in_spans,
            mask_tool=self._telemetry_policy.mask_tool_names_in_spans,
        )
        return types.TelemetrySpansListResult(spans=list_span_views(spans))

    async def _telemetry_span_read(self, span: types.SpanScope) -> types.TelemetrySpanReadResult:
        events = await self.store.load_telemetry(span.system_session_id, span.run_id, kinds={"span"})
        spans = build_span_index(
            events,
            mask_model=self._telemetry_policy.mask_model_in_spans,
            mask_tool=self._telemetry_policy.mask_tool_names_in_spans,
        )
        return types.TelemetrySpanReadResult(span=get_span_view(spans, span.span_id))

    async def _telemetry_payload_read(self, span: types.SpanScope) -> types.TelemetryPayloadReadResult:
        env = await self.store.get_span_payload(span.system_session_id, span.run_id, span.span_id)
        if not env:
            return types.TelemetryPayloadReadResult(payload=None)
        sanitized = sanitize_envelope_dict(env.model_dump(), self._telemetry_policy)
        return types.TelemetryPayloadReadResult(payload=SpanPayloadEnvelope.model_validate(sanitized))

    async def _conversations_channels_list(self, runs: types.RunsScope) -> types.ChannelsListResult:
        # Gate index listing if entirely private-only
        if is_private_only_conversations_selector(self._conversation_policy, channel=None):
            raise PermissionError(
                "Listing denied: channels index is private under current conversation policy."
            )
        metas = await self.store.list_channel_meta(runs.system_session_id, runs.run_id)
        return types.ChannelsListResult(channels=metas)

    async def _conversations_channel_read(
        self, channel: types.ChannelScope
    ) -> types.ChannelReadResult:
        # Deny if this channel would be private-only under policy
        if is_private_only_conversations_selector(self._conversation_policy, channel=channel.channel):
            raise PermissionError("Read denied: channel is private under current conversation policy.")

        # meta
        metas = await self.store.list_channel_meta(channel.system_session_id, channel.run_id)
        meta = next((m for m in metas if m.id == channel.channel), ChannelMeta(id=channel.channel))
        # raw items (per channel), then sanitize
        raw_items = await self.store.load_channel_items(channel.system_session_id, channel.run_id, channel=channel.channel)
        sanitized = sanitize_conversation_items(raw_items, self._conversation_policy)
        ch = ChannelView(meta=meta, items=sanitized)
        return types.ChannelReadResult(channel=ch)

    async def _system_sessions_list(self) -> types.SystemSessionsListResult:
        sess = await self.store.list_system_sessions()
        return types.SystemSessionsListResult(sessions=sess)

    async def _system_session_read(self, session: types.SystemSessionScope) -> types.SystemSessionReadResult:
        s = await self.store.get_system_session(session.system_session_id)
        return types.SystemSessionReadResult(session=s if s else None)

    def get_context(self) -> Context[ServerSession, LifespanResultT, Request]:
        """
        Returns a Context object. Note that the context will only be valid
        during a request; outside a request, most methods will error.
        """
        try:
            request_context = self._wrp_server.request_context
        except LookupError:
            request_context = None
        return Context(request_context=request_context, wrp=self)

    @property
    def store(self) -> Store:
        """Access the configured run store."""
        return self._store

    @property
    def telemetry_policy(self) -> TelemetryResourcePolicy:
        return self._telemetry_policy

    def set_telemetry_resource_policy(self, policy: TelemetryResourcePolicy) -> None:
        self._telemetry_policy = policy

    @property
    def conversation_policy(self) -> ConversationResourcePolicy:
        return self._conversation_policy

    def set_conversation_resource_policy(self, policy: ConversationResourcePolicy) -> None:
        self._conversation_policy = policy

    # ---- Global default seeding (fallback used when a workflow doesn't set one) ----
    @property
    def default_seeding(self) -> WorkflowConversationSeeding | None:
        return self._default_seeding

    def set_default_seeding(self, seeding: WorkflowConversationSeeding | None) -> None:
        self._default_seeding = seeding

    async def _list_workflows(self, _req: types.ListWorkflowsRequest,) -> types.ListWorkflowsResult:
        """Return public descriptors for all registered workflows (no pagination)."""
        return types.ListWorkflowsResult(workflows=self._workflow_manager.list_descriptors())

    async def _run_workflow(self, name: str, wf_input: dict[str, Any]) -> RunWorkflowResult:
        """Run a workflow by name with a dict payload matching its WorkflowInput.

        Note: This uses the current request context if available. Outside a request,
        the context is still constructed by `get_context()`, but most MCP-bound
        features (like logging or progress reporting) will raise an error if called.
        """
        context = self.get_context()
        # Ensure effective settings are hydrated from durable store before executing
        await self._workflow_manager.load_persisted_settings_if_needed(name)
        return await self._workflow_manager.run(name, wf_input, context)

    async def _list_resources(self) -> list[types.Resource]:
        """List all available resources."""

        resources = self._resource_manager.list_resources()
        return [
            types.Resource(
                uri=resource.uri,
                name=resource.name or "",
                title=resource.title,
                description=resource.description,
                mimeType=resource.mime_type,
                icons=resource.icons,
            )
            for resource in resources
        ]

    async def _list_resource_templates(self) -> list[types.ResourceTemplate]:
        templates = self._resource_manager.list_templates()
        return [
            types.ResourceTemplate(
                uriTemplate=template.uri_template,
                name=template.name,
                title=template.title,
                description=template.description,
                mimeType=template.mime_type,
                icons=template.icons,
            )
            for template in templates
        ]

    async def _read_resource(self, uri: AnyUrl | str) -> Iterable[ReadResourceContents]:
        """Read a resource by URI."""

        context = self.get_context()
        resource = await self._resource_manager.get_resource(uri, context=context)
        if not resource:
            raise ResourceError(f"Unknown resource: {uri}")

        try:
            content = await resource.read()
            return [ReadResourceContents(content=content, mime_type=resource.mime_type)]
        except Exception as e:
            logger.exception(f"Error reading resource {uri}")
            raise ResourceError(str(e))

    def add_workflow(
        self,
        fn: AnyFunction,
        *,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        input_model: type[WorkflowInput] | None = None,
        output_model: type[WorkflowOutput] | None = None,
        icons: list[types.Icon] | None = None,
        seeding: WorkflowConversationSeeding | None = None,
        input_limit_bytes: int | None = None,
        settings_default: WorkflowSettings | None = None,
        settings_allow_override: bool = True,
    ) -> None:
        """Register a workflow function.

        The workflow function should look like:

            async def my_flow(wf_input: MyInput, ctx: Context) -> MyOutput: ...

        Where:
          - `MyInput` subclasses `WorkflowInput`
          - `MyOutput` subclasses `WorkflowOutput`
          - `ctx` is optional; if present, it must be type-annotated as `Context`
        """
        self._workflow_manager.add_workflow(
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

    def workflow(
        self,
        *,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        input_model: type[WorkflowInput] | None = None,
        output_model: type[WorkflowOutput] | None = None,
        icons: list[types.Icon] | None = None,
        seeding: WorkflowConversationSeeding | None = None,
        input_limit_bytes: int | None = None,
        settings_default: WorkflowSettings | None = None,
        settings_allow_override: bool = True,
    ) -> Callable[[AnyFunction], AnyFunction]:
        """Decorator to register a workflow.

        Example:
            class MyIn(WorkflowInput):
                query: str

            class MyOut(WorkflowOutput):
                answer: str

            @server.workflow(name="qa", title="Q&A")
            async def qa_flow(wf_input: MyIn, ctx: Context) -> MyOut:
                ...
                return MyOut(answer="...")

        """
        if callable(name):
            raise TypeError(
                "The @workflow decorator was used incorrectly. Did you forget to call it? "
                "Use @workflow() instead of @workflow"
            )

        def decorator(fn: AnyFunction) -> AnyFunction:
            self.add_workflow(
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
            return fn

        return decorator

    def add_resource(self, resource: Resource) -> None:
        """Add a resource to the server.

        Args:
            resource: A Resource instance to add
        """
        self._resource_manager.add_resource(resource)

    def resource(
        self,
        uri: str,
        *,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        mime_type: str | None = None,
        icons: list[types.Icon] | None = None,
    ) -> Callable[[AnyFunction], AnyFunction]:
        """Decorator to register a function as a resource.

        The function will be called when the resource is read to generate its content.
        The function can return:
        - str for text content
        - bytes for binary content
        - other types will be converted to JSON

        If the URI contains parameters (e.g. "resource://{param}") or the function
        has parameters, it will be registered as a template resource.

        Args:
            uri: URI for the resource (e.g. "resource://my-resource" or "resource://{param}")
            name: Optional name for the resource
            title: Optional human-readable title for the resource
            description: Optional description of the resource
            mime_type: Optional MIME type for the resource

        Example:
            @server.resource("resource://my-resource")
            def get_data() -> str:
                return "Hello, world!"

            @server.resource("resource://my-resource")
            async get_data() -> str:
                data = await fetch_data()
                return f"Hello, world! {data}"

            @server.resource("resource://{city}/weather")
            def get_weather(city: str) -> str:
                return f"Weather for {city}"

            @server.resource("resource://{city}/weather")
            async def get_weather(city: str) -> str:
                data = await fetch_weather(city)
                return f"Weather for {city}: {data}"
        """
        # Check if user passed function directly instead of calling decorator
        if callable(uri):
            raise TypeError(
                "The @resource decorator was used incorrectly. "
                "Did you forget to call it? Use @resource('uri') instead of @resource"
            )

        def decorator(fn: AnyFunction) -> AnyFunction:
            # Check if this should be a template
            sig = inspect.signature(fn)
            has_uri_params = "{" in uri and "}" in uri
            has_func_params = bool(sig.parameters)

            if has_uri_params or has_func_params:
                # Check for Context parameter to exclude from validation
                context_param = find_context_parameter(fn)

                # Validate that URI params match function params (excluding context)
                uri_params = set(re.findall(r"{(\w+)}", uri))
                # We need to remove the context_param from the resource function if
                # there is any.
                func_params = {p for p in sig.parameters.keys() if p != context_param}

                if uri_params != func_params:
                    raise ValueError(
                        f"Mismatch between URI parameters {uri_params} and function parameters {func_params}"
                    )

                # Register as template
                self._resource_manager.add_template(
                    fn=fn,
                    uri_template=uri,
                    name=name,
                    title=title,
                    description=description,
                    mime_type=mime_type,
                    icons=icons,
                )
            else:
                # Register as regular resource
                resource = FunctionResource.from_function(
                    fn=fn,
                    uri=uri,
                    name=name,
                    title=title,
                    description=description,
                    mime_type=mime_type,
                    icons=icons,
                )
                self.add_resource(resource)
            return fn

        return decorator

    # ---- Provider / Agent settings registration APIs ---------------------

    def register_provider_settings(
        self,
        name: str,
        default: ProviderSettings,
        *,
        allow_override: bool = True,
    ) -> None:
        """
        Register provider-level settings defaults.

        These are global (non-workflow) and typically include API keys and
        other provider configuration such as endpoint URLs.
        """
        self._provider_settings_registry.register(name, default, allow_override=allow_override)

    def get_provider_settings(self, name: str) -> ProviderSettings | None:
        """
        Return a deep copy of the effective ProviderSettings for the given provider.
        """
        cfg = self._provider_settings_registry.get(name)
        return cfg.model_copy(deep=True) if cfg is not None else None

    def register_agent_settings(
        self,
        name: str,
        default: AgentSettings,
        *,
        allow_override: bool = True,
    ) -> None:
        """
        Register agent-level settings defaults.

        These are global configuration blobs keyed by agent name and usually
        reference a provider via `provider_name`.
        """
        self._agent_settings_registry.register(name, default, allow_override=allow_override)

    def get_agent_settings(self, name: str) -> AgentSettings | None:
        """
        Return a deep copy of the effective AgentSettings for the given agent.
        """
        cfg = self._agent_settings_registry.get(name)
        return cfg.model_copy(deep=True) if cfg is not None else None

    def custom_route(
        self,
        path: str,
        methods: list[str],
        name: str | None = None,
        include_in_schema: bool = True,
    ):
        """
        Decorator to register a custom HTTP route on the WRP server.

        Allows adding arbitrary HTTP endpoints outside the standard WRP protocol,
        which can be useful for OAuth callbacks, health checks, or admin APIs.
        The handler function must be an async function that accepts a Starlette
        Request and returns a Response.

        Args:
            path: URL path for the route (e.g., "/oauth/callback")
            methods: List of HTTP methods to support (e.g., ["GET", "POST"])
            name: Optional name for the route (to reference this route with
                  Starlette's reverse URL lookup feature)
            include_in_schema: Whether to include in OpenAPI schema, defaults to True

        Example:
            @server.custom_route("/health", methods=["GET"])
            async def health_check(request: Request) -> Response:
                return JSONResponse({"status": "ok"})
        """

        def decorator(
            func: Callable[[Request], Awaitable[Response]],
        ) -> Callable[[Request], Awaitable[Response]]:
            self._custom_starlette_routes.append(
                Route(
                    path,
                    endpoint=func,
                    methods=methods,
                    name=name,
                    include_in_schema=include_in_schema,
                )
            )
            return func

        return decorator

    async def run_stdio_async(self) -> None:
        """Run the server using stdio transport."""
        # Ensure provider/agent settings are hydrated from the store before serving.
        await hydrate_provider_and_agent_settings(self)
        async with stdio_server() as (read_stream, write_stream):
            await self._wrp_server.run(
                read_stream,
                write_stream,
                self._wrp_server.create_initialization_options(),
            )

    async def run_sse_async(self, mount_path: str | None = None) -> None:
        """Run the server using SSE transport."""
        import uvicorn

        # Hydrate settings before creating the app / accepting connections.
        await hydrate_provider_and_agent_settings(self)
        starlette_app = self.sse_app(mount_path)

        config = uvicorn.Config(
            starlette_app,
            host=self.settings.host,
            port=self.settings.port,
            log_level=self.settings.log_level.lower(),
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def run_streamable_http_async(self) -> None:
        """Run the server using StreamableHTTP transport."""
        import uvicorn

        # Hydrate settings before creating the app / accepting connections.
        await hydrate_provider_and_agent_settings(self)
        starlette_app = self.streamable_http_app()

        config = uvicorn.Config(
            starlette_app,
            host=self.settings.host,
            port=self.settings.port,
            log_level=self.settings.log_level.lower(),
        )
        server = uvicorn.Server(config)
        await server.serve()

    def _normalize_path(self, mount_path: str, endpoint: str) -> str:
        """
        Combine mount path and endpoint to return a normalized path.

        Args:
            mount_path: The mount path (e.g. "/github" or "/")
            endpoint: The endpoint path (e.g. "/messages/")

        Returns:
            Normalized path (e.g. "/github/messages/")
        """
        # Special case: root path
        if mount_path == "/":
            return endpoint

        # Remove trailing slash from mount path
        if mount_path.endswith("/"):
            mount_path = mount_path[:-1]

        # Ensure endpoint starts with slash
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint

        # Combine paths
        return mount_path + endpoint

    def sse_app(self, mount_path: str | None = None) -> Starlette:
        """Return an instance of the SSE server app."""
        # Update mount_path in settings if provided
        if mount_path is not None:
            self.settings.mount_path = mount_path

        # Create normalized endpoint considering the mount path
        normalized_message_endpoint = self._normalize_path(self.settings.mount_path, self.settings.message_path)

        # Set up auth context and dependencies

        sse = SseServerTransport(
            normalized_message_endpoint,
            security_settings=self.settings.transport_security,
        )

        async def handle_sse(scope: Scope, receive: Receive, send: Send):
            # Add client ID from auth context into request context if available

            async with sse.connect_sse(
                scope,
                receive,
                send,
            ) as streams:
                await self._wrp_server.run(
                    streams[0],
                    streams[1],
                    self._wrp_server.create_initialization_options(),
                )
            return Response()

        # Create routes
        routes: list[Route | Mount] = []
        middleware: list[Middleware] = []
        required_scopes = []

        # Set up auth if configured
        if self.settings.auth:
            required_scopes = self.settings.auth.required_scopes or []

            # Add auth middleware if token verifier is available
            if self._token_verifier:
                middleware = [
                    # extract auth info from request (but do not require it)
                    Middleware(
                        AuthenticationMiddleware,
                        backend=BearerAuthBackend(self._token_verifier),
                    ),
                    # Add the auth context middleware to store
                    # authenticated user in a contextvar
                    Middleware(AuthContextMiddleware),
                ]

            # Add auth endpoints if auth server provider is configured
            if self._auth_server_provider:
                from wrp.server.auth.routes import create_auth_routes

                routes.extend(
                    create_auth_routes(
                        provider=self._auth_server_provider,
                        issuer_url=self.settings.auth.issuer_url,
                        service_documentation_url=self.settings.auth.service_documentation_url,
                        client_registration_options=self.settings.auth.client_registration_options,
                        revocation_options=self.settings.auth.revocation_options,
                    )
                )

        # When auth is configured, require authentication
        if self._token_verifier:
            # Determine resource metadata URL
            resource_metadata_url = None
            if self.settings.auth and self.settings.auth.resource_server_url:
                from pydantic import AnyHttpUrl

                resource_metadata_url = AnyHttpUrl(
                    str(self.settings.auth.resource_server_url).rstrip("/") + "/.well-known/oauth-protected-resource"
                )

            # Auth is enabled, wrap the endpoints with RequireAuthMiddleware
            routes.append(
                Route(
                    self.settings.sse_path,
                    endpoint=RequireAuthMiddleware(handle_sse, required_scopes, resource_metadata_url),
                    methods=["GET"],
                )
            )
            routes.append(
                Mount(
                    self.settings.message_path,
                    app=RequireAuthMiddleware(sse.handle_post_message, required_scopes, resource_metadata_url),
                )
            )
        else:
            # Auth is disabled, no need for RequireAuthMiddleware
            # Since handle_sse is an ASGI app, we need to create a compatible endpoint
            async def sse_endpoint(request: Request) -> Response:
                # Convert the Starlette request to ASGI parameters
                return await handle_sse(request.scope, request.receive, request._send)  # type: ignore[reportPrivateUsage]

            routes.append(
                Route(
                    self.settings.sse_path,
                    endpoint=sse_endpoint,
                    methods=["GET"],
                )
            )
            routes.append(
                Mount(
                    self.settings.message_path,
                    app=sse.handle_post_message,
                )
            )
        # Add protected resource metadata endpoint if configured as RS
        if self.settings.auth and self.settings.auth.resource_server_url:
            from wrp.server.auth.routes import create_protected_resource_routes

            routes.extend(
                create_protected_resource_routes(
                    resource_url=self.settings.auth.resource_server_url,
                    authorization_servers=[self.settings.auth.issuer_url],
                    scopes_supported=self.settings.auth.required_scopes,
                )
            )

        # mount these routes last, so they have the lowest route matching precedence
        routes.extend(self._custom_starlette_routes)

        # Create Starlette app with routes and middleware
        return Starlette(debug=self.settings.debug, routes=routes, middleware=middleware)

    def streamable_http_app(self) -> Starlette:
        """Return an instance of the StreamableHTTP server app."""
        # Create session manager on first call (lazy initialization)
        if self._session_manager is None:
            self._session_manager = StreamableHTTPSessionManager(
                app=cast("MCPServer[Any, Any]", self._wrp_server),  # typing-only cast
                event_store=self._event_store,
                json_response=self.settings.json_response,
                stateless=self.settings.stateless_http,  # Use the stateless setting
                security_settings=self.settings.transport_security,
            )

        # Create the ASGI handler
        streamable_http_app = StreamableHTTPASGIApp(self._session_manager)

        # Create routes
        routes: list[Route | Mount] = []
        middleware: list[Middleware] = [
            Middleware(
                HeaderCompatMiddleware,
                internal_protocol_value=MCP_DEFAULT_PROTOCOL,  # feed MCP-supported value internally
                wrp_supported_versions=WRP_SUPPORTED_VERSIONS,
                require_wrp_header=True,
            )
        ]
        required_scopes = []

        # Set up auth if configured
        if self.settings.auth:
            required_scopes = self.settings.auth.required_scopes or []

            # Add auth middleware if token verifier is available
            if self._token_verifier:
                middleware.extend(
                    [
                        Middleware(
                            AuthenticationMiddleware,
                            backend=BearerAuthBackend(self._token_verifier),
                        ),
                        Middleware(AuthContextMiddleware),
                    ]
                )

            # Add auth endpoints if auth server provider is configured
            if self._auth_server_provider:
                from wrp.server.auth.routes import create_auth_routes

                routes.extend(
                    create_auth_routes(
                        provider=self._auth_server_provider,
                        issuer_url=self.settings.auth.issuer_url,
                        service_documentation_url=self.settings.auth.service_documentation_url,
                        client_registration_options=self.settings.auth.client_registration_options,
                        revocation_options=self.settings.auth.revocation_options,
                    )
                )

        # Set up routes with or without auth
        if self._token_verifier:
            # Determine resource metadata URL
            resource_metadata_url = None
            if self.settings.auth and self.settings.auth.resource_server_url:
                from pydantic import AnyHttpUrl

                resource_metadata_url = AnyHttpUrl(
                    str(self.settings.auth.resource_server_url).rstrip("/") + "/.well-known/oauth-protected-resource"
                )

            routes.append(
                Route(
                    self.settings.streamable_http_path,
                    endpoint=RequireAuthMiddleware(streamable_http_app, required_scopes, resource_metadata_url),
                )
            )
        else:
            # Auth is disabled, no wrapper needed
            routes.append(
                Route(
                    self.settings.streamable_http_path,
                    endpoint=streamable_http_app,
                )
            )

        # Add protected resource metadata endpoint if configured as RS
        if self.settings.auth and self.settings.auth.resource_server_url:
            from mcp.server.auth.handlers.metadata import ProtectedResourceMetadataHandler
            from wrp.server.auth.routes import cors_middleware
            from mcp.shared.auth import ProtectedResourceMetadata

            protected_resource_metadata = ProtectedResourceMetadata(
                resource=self.settings.auth.resource_server_url,
                authorization_servers=[self.settings.auth.issuer_url],
                scopes_supported=self.settings.auth.required_scopes,
            )
            routes.append(
                Route(
                    "/.well-known/oauth-protected-resource",
                    endpoint=cors_middleware(
                        ProtectedResourceMetadataHandler(protected_resource_metadata).handle,
                        ["GET", "OPTIONS"],
                    ),
                    methods=["GET", "OPTIONS"],
                )
            )

        routes.extend(self._custom_starlette_routes)

        return Starlette(
            debug=self.settings.debug,
            routes=routes,
            middleware=middleware,
            lifespan=lambda app: self.session_manager.run(),
        )

    # ---- Workflow settings protocol handlers (runtime mechanics) ----
    async def _wf_settings_read(self, workflow: str) -> types.WorkflowSettingsReadResult:
        """Return effective settings + flags for a workflow."""
        await self._workflow_manager.load_persisted_settings_if_needed(workflow)
        cur = self._workflow_manager.get_settings(workflow)
        wf = self._workflow_manager.get(workflow)
        if cur is None:
            return types.WorkflowSettingsReadResult(
                values=None,
                overridden=self._workflow_manager.settings_overridden(workflow),
                allowOverride=bool(wf.settings_allow_override) if wf else False,
                locked=None,
            )
        return types.WorkflowSettingsReadResult(
            values=cur.model_dump(),
            overridden=self._workflow_manager.settings_overridden(workflow),
            allowOverride=bool(wf.settings_allow_override) if wf else False,
            locked=list(getattr(cur.__class__, "locked", set())),
        )

    async def _wf_settings_schema(self, workflow: str) -> types.WorkflowSettingsSchemaResult:
        """Return JSON schema for a workflow's settings (if defined)."""
        wf = self._workflow_manager.get(workflow)
        if not wf or wf.settings_default is None:
            return types.WorkflowSettingsSchemaResult(jsonSchema=None)
        return types.WorkflowSettingsSchemaResult(
            jsonSchema=wf.settings_default.__class__.model_json_schema(by_alias=True)
        )

    async def _wf_settings_update(self, workflow: str, values: dict[str, Any]) -> types.WorkflowSettingsReadResult:
        """Merge/update workflow settings and return the effective view."""
        inst = await self._workflow_manager.update_settings(workflow, values or {})
        wf = self._workflow_manager.get(workflow)
        return types.WorkflowSettingsReadResult(
            values=inst.model_dump(),
            overridden=True,
            allowOverride=bool(wf.settings_allow_override) if wf else False,
            locked=list(getattr(inst.__class__, "locked", set())),
        )

    # ---- Provider settings protocol handlers ------------------------------
    async def _provider_settings_read(self, provider: str) -> types.ProviderSettingsReadResult:
        """Return effective settings + flags for a provider."""
        await self._provider_settings_registry.load_persisted_if_needed(provider)
        cur = self._provider_settings_registry.get(provider)
        if cur is None:
            return types.ProviderSettingsReadResult(
                values=None,
                overridden=self._provider_settings_registry.settings_overridden(provider),
                allowOverride=self._provider_settings_registry.allow_override(provider),
                locked=None,
                secrets=None,
            )
        values, secrets = self._provider_settings_registry.mask_values(cur)
        secrets_view = (
            {k: types.ProviderSecretSummary(hasValue=v["hasValue"]) for k, v in secrets.items()}
            if secrets
            else None
        )
        return types.ProviderSettingsReadResult(
            values=values,
            overridden=self._provider_settings_registry.settings_overridden(provider),
            allowOverride=self._provider_settings_registry.allow_override(provider),
            locked=list(getattr(cur.__class__, "locked", set())),
            secrets=secrets_view,
        )

    async def _provider_settings_schema(self, provider: str) -> types.ProviderSettingsSchemaResult:
        """Return JSON schema for a provider's settings (if defined)."""
        schema = self._provider_settings_registry.schema_for(provider)
        return types.ProviderSettingsSchemaResult(jsonSchema=schema)

    async def _provider_settings_update(
        self,
        provider: str,
        values: dict[str, Any],
    ) -> types.ProviderSettingsReadResult:
        """Merge/update provider settings and return the effective view."""
        inst = await self._provider_settings_registry.update(provider, values or {})
        masked_values, secrets = self._provider_settings_registry.mask_values(inst)
        secrets_view = (
            {k: types.ProviderSecretSummary(hasValue=v["hasValue"]) for k, v in secrets.items()}
            if secrets
            else None
        )
        return types.ProviderSettingsReadResult(
            values=masked_values,
            overridden=True,
            allowOverride=self._provider_settings_registry.allow_override(provider),
            locked=list(getattr(inst.__class__, "locked", set())),
            secrets=secrets_view,
        )

    # ---- Agent settings protocol handlers --------------------------------
    async def _agent_settings_read(self, agent: str) -> types.AgentSettingsReadResult:
        """Return effective settings + flags for an agent."""
        await self._agent_settings_registry.load_persisted_if_needed(agent)
        cur = self._agent_settings_registry.get(agent)
        if cur is None:
            return types.AgentSettingsReadResult(
                values=None,
                overridden=self._agent_settings_registry.settings_overridden(agent),
                allowOverride=self._agent_settings_registry.allow_override(agent),
                locked=None,
            )
        return types.AgentSettingsReadResult(
            values=cur.model_dump(),
            overridden=self._agent_settings_registry.settings_overridden(agent),
            allowOverride=self._agent_settings_registry.allow_override(agent),
            locked=list(getattr(cur.__class__, "locked", set())),
        )

    async def _agent_settings_schema(self, agent: str) -> types.AgentSettingsSchemaResult:
        """Return JSON schema for an agent's settings (if defined)."""
        schema = self._agent_settings_registry.schema_for(agent)
        return types.AgentSettingsSchemaResult(jsonSchema=schema)

    async def _agent_settings_update(
        self,
        agent: str,
        values: dict[str, Any],
    ) -> types.AgentSettingsReadResult:
        """Merge/update agent settings and return the effective view."""
        inst = await self._agent_settings_registry.update(agent, values or {})
        return types.AgentSettingsReadResult(
            values=inst.model_dump(),
            overridden=True,
            allowOverride=self._agent_settings_registry.allow_override(agent),
            locked=list(getattr(inst.__class__, "locked", set())),
        )


class StreamableHTTPASGIApp:
    """
    ASGI application for Streamable HTTP server transport.
    """

    def __init__(self, session_manager: StreamableHTTPSessionManager):
        self.session_manager = session_manager

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.session_manager.handle_request(scope, receive, send)


class Context(BaseModel, Generic[ServerSessionT, LifespanContextT, RequestT]):
    """Context object providing access to WRP capabilities.

    This provides a cleaner interface to WRP's RequestContext functionality.
    It gets injected into workflow and resource functions that request it via type hints.

    To use context in a resource or workflow function, add a parameter with the Context type annotation:

    ```python
    @server.resource("resource://my-data")
    async def get_my_data(ctx: Context) -> str:
        # Log messages to the client
        await ctx.info(f"Generating data for request {ctx.request_id}")
        await ctx.debug("Debug info")

        # Report progress
        await ctx.report_progress(50, 100)

        # Access other resources
        other_data_contents = await ctx.read_resource("resource://other-data")
        # other_data_contents is an iterable of ReadResourceContents

        # Get request info
        request_id = ctx.request_id
        client_id = ctx.client_id

        return f"Data for {client_id}"
    ```

    The context parameter name can be anything as long as it's annotated with Context.
    The context is optional - functions that don't need it can omit the parameter.
    """

    _request_context: RequestContext[ServerSessionT, LifespanContextT, RequestT] | None
    _wrp: WRP | None
    _run: RunBindings | None = None

    def __init__(
        self,
        *,
        request_context: (RequestContext[ServerSessionT, LifespanContextT, RequestT] | None) = None,
        wrp: WRP | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._request_context = request_context
        self._wrp = wrp

    @property
    def wrp(self) -> WRP:
        """Access to the WRP server."""
        if self._wrp is None:
            raise ValueError("Context is not available outside of a request")
        return self._wrp

    @property
    def run(self) -> RunBindings:
        """Per-run APIs (conversations/telemetry). Only valid during a workflow run."""
        if self._run is None:
            raise ValueError("Run context is not available outside of a workflow run")
        return self._run

    # internal: attached by WorkflowManager when a run starts
    def _attach_run(self, bindings: RunBindings) -> None:
        self._run = bindings

    @property
    def request_context(
        self,
    ) -> RequestContext[ServerSessionT, LifespanContextT, RequestT]:
        """Access to the underlying request context."""
        if self._request_context is None:
            raise ValueError("Context is not available outside of a request")
        return self._request_context

    async def report_progress(self, progress: float, total: float | None = None, message: str | None = None) -> None:
        """Report progress for the current operation.

        Args:
            progress: Current progress value e.g. 24
            total: Optional total value e.g. 100
            message: Optional message e.g. Starting render...
        """
        progress_token = self.request_context.meta.progressToken if self.request_context.meta else None

        if progress_token is None:
            return

        await self.request_context.session.send_progress_notification(
            progress_token=progress_token,
            progress=progress,
            total=total,
            message=message,
        )

    async def read_resource(self, uri: str | AnyUrl) -> Iterable[ReadResourceContents]:
        """Read a resource by URI.

        Args:
            uri: Resource URI to read

        Returns:
            The resource content as either text or bytes
        """
        assert self._wrp is not None, "Context is not available outside of a request"
        return await self._wrp._read_resource(uri)

    async def elicit(
        self,
        message: str,
        schema: type[ElicitSchemaModelT],
    ) -> ElicitationResult[ElicitSchemaModelT]:
        """Elicit information from the client/user.

        This method can be used to interactively ask for additional information from the
        client within a workflow's execution. The client might display the message to the
        user and collect a response according to the provided schema. Or in case a
        client is an agent, it might decide how to handle the elicitation -- either by asking
        the user or automatically generating a response.

        Args:
            schema: A Pydantic model class defining the expected response structure, according to the specification,
                    only primive types are allowed.
            message: Optional message to present to the user. If not provided, will use
                    a default message based on the schema

        Returns:
            An ElicitationResult containing the action taken and the data if accepted

        Note:
            Check the result.action to determine if the user accepted, declined, or cancelled.
            The result.data will only be populated if action is "accept" and validation succeeded.
        """

        return await elicit_with_validation(
            session=self.request_context.session, message=message, schema=schema, related_request_id=self.request_id
        )

    async def log(
        self,
        level: Literal["debug", "info", "warning", "error"],
        message: str,
        *,
        logger_name: str | None = None,
    ) -> None:
        """Send a log message to the client.

        Args:
            level: Log level (debug, info, warning, error)
            message: Log message
            logger_name: Optional logger name
            **extra: Additional structured data to include
        """
        await self.request_context.session.send_log_message(
            level=level,
            data=message,
            logger=logger_name,
            related_request_id=self.request_id,
        )

    @property
    def client_id(self) -> str | None:
        """Get the client ID if available."""
        return getattr(self.request_context.meta, "client_id", None) if self.request_context.meta else None

    @property
    def system_session_id(self) -> str | None:
        """Get the system session ID if available."""
        req = getattr(self.request_context, "request", None)
        if isinstance(req, dict):
            wrp_ns = req.get("wrp") or {}
            val = wrp_ns.get("system_session")
            if isinstance(val, str) and val:
                return val
        return None

    @property
    def request_id(self) -> str:
        """Get the unique ID for this request."""
        return str(self.request_context.request_id)

    @property
    def session(self):
        """Access to the underlying session for advanced usage."""
        return self.request_context.session

    # Convenience methods for common log levels
    async def debug(self, message: str, **extra: Any) -> None:
        """Send a debug log message."""
        await self.log("debug", message, **extra)

    async def info(self, message: str, **extra: Any) -> None:
        """Send an info log message."""
        await self.log("info", message, **extra)

    async def warning(self, message: str, **extra: Any) -> None:
        """Send a warning log message."""
        await self.log("warning", message, **extra)

    async def error(self, message: str, **extra: Any) -> None:
        """Send an error log message."""
        await self.log("error", message, **extra)

    # ---- Workflow settings convenience ----
    def get_workflow_settings(self, name: str | None = None) -> WorkflowSettings | None:
        """
        Return the effective non-run WorkflowSettings instance for the given workflow name.
        If name is omitted, uses the current run's workflow.
        """
        wf_name = name or (self._run.workflow_name if self._run else None)  # type: ignore[attr-defined]
        if not wf_name:
            raise ValueError("No workflow is active, and no workflow name was provided")
        cfg = self.wrp._workflow_manager.get_settings(wf_name)
        # Hand workflows a deep copy so in-process mutations don't persist.
        return cfg.model_copy(deep=True) if cfg is not None else None

    def get_provider_settings(self, name: str) -> ProviderSettings | None:
        """
        Return the effective ProviderSettings for the given provider name.
        """
        cfg = self.wrp.get_provider_settings(name)
        return cfg

    def get_agent_settings(self, name: str) -> AgentSettings | None:
        """
        Return the effective AgentSettings for the given agent name.
        """
        cfg = self.wrp.get_agent_settings(name)
        return cfg