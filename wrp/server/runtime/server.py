# wrp/server/runtime/server.py
from __future__ import annotations as _annotations

import inspect
import re
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Iterable, Sequence
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
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send

from mcp.server.auth.middleware.auth_context import AuthContextMiddleware
from mcp.server.auth.middleware.bearer_auth import BearerAuthBackend, RequireAuthMiddleware
from mcp.server.auth.provider import OAuthAuthorizationServerProvider, ProviderTokenVerifier, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp.exceptions import ResourceError
from mcp.server.fastmcp.resources import FunctionResource, Resource, ResourceManager
from mcp.server.fastmcp.utilities.context_injection import find_context_parameter
from mcp.server.fastmcp.utilities.logging import configure_logging, get_logger
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.sse import SseServerTransport
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http import EventStore
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import AnyFunction, Icon
from mcp.types import Resource as MCPResource
from mcp.types import ResourceTemplate as MCPResourceTemplate
from mcp.types import DEFAULT_NEGOTIATED_VERSION as MCP_DEFAULT_PROTOCOL

from wrp.server.elicitation import ElicitationResult, ElicitSchemaModelT, elicit_with_validation
from wrp.server.lowlevel.server import LifespanResultT
from wrp.server.lowlevel.server import Server as WRPServer
from wrp.server.lowlevel.server import lifespan as default_lifespan
from wrp.server.middleware.header_compat import HeaderCompatMiddleware
from wrp.server.session import ServerSession, ServerSessionT
from wrp.server.runtime.limits import DEFAULT_GLOBAL_INPUT_LIMIT_BYTES
from wrp.server.runtime.runs.bindings import RunBindings
from wrp.server.runtime.store.base import Store
from wrp.server.runtime.store.stores.memory_store import InMemoryStore
from wrp.server.runtime.telemetry.privacy.guards import is_private_only_span_payload_uri
from wrp.server.runtime.conversations.privacy.guards import (
    is_private_only_conversations_uri,
)
from wrp.server.runtime.telemetry.privacy.policy import TelemetryResourcePolicy
from wrp.server.runtime.telemetry.privacy.redaction import sanitize_envelope_dict
from wrp.server.runtime.conversations.privacy.policy import ConversationResourcePolicy
from wrp.server.runtime.conversations.privacy.redaction import sanitize_conversation_items
from wrp.server.runtime.telemetry.views import (
    build_span_index,
    serialize_span_detail,
    serialize_span_list,
)
from wrp.server.runtime.workflows import WorkflowManager
from wrp.server.runtime.conversations.seeding import WorkflowConversationSeeding
from wrp.server.runtime.workflows.types import RunWorkflowResult, WorkflowInput, WorkflowOutput
from wrp.server.runtime.workflows.settings import WorkflowSettings
from wrp.shared.context import LifespanContextT, RequestContext, RequestT
from wrp.shared.version import SUPPORTED_PROTOCOL_VERSIONS as WRP_SUPPORTED_VERSIONS
from wrp.types import ListWorkflowsResult

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
        icons: list[Icon] | None = None,
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
        _settings_kwargs = dict(
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
            # TODO(Marcelo): It seems there's a type mismatch between the lifespan type from an WRP and Server.
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
        # uri -> WeakSet[ServerSession]
        self._resource_subscriptions: dict[str, "weakref.WeakSet[ServerSession]"] = {}

        # Set up WRP protocol handlers
        self._setup_handlers()
        self._register_builtin_run_resources()
        self._register_builtin_workflow_settings_endpoints()

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
    def icons(self) -> list[Icon] | None:
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
        self._wrp_server.list_resources()(self.list_resources)
        self._wrp_server.read_resource()(self.read_resource)
        self._wrp_server.list_resource_templates()(self.list_resource_templates)
        # subscriptions
        self._wrp_server.subscribe_resource()(self._subscribe_resource)
        self._wrp_server.unsubscribe_resource()(self._unsubscribe_resource)

        self._wrp_server.list_workflows()(self.list_workflows)
        self._wrp_server.run_workflow(validate_input=True)(self.run_workflow)

    async def _subscribe_resource(self, uri: AnyUrl) -> None:
        """Remember that the *current* session wants updates for this uri."""
        ctx = self.get_context()
        # Block subscriptions to payload resources that are private-only per policy
        if await is_private_only_span_payload_uri(str(uri), self._telemetry_policy, self._store):
            raise ResourceError("Resource is private-only by policy; subscriptions are not allowed")
        # Block subscriptions to conversations resources that are private-only per policy
        if await is_private_only_conversations_uri(str(uri), self._conversation_policy, self._store):
            raise ResourceError("Resource is private-only by policy; subscriptions are not allowed")
        sess = ctx.request_context.session
        bucket = self._resource_subscriptions.setdefault(str(uri), weakref.WeakSet())
        bucket.add(sess)

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

    async def notify_resource_updated(self, uri: str | AnyUrl) -> None:
        """Push resource/updated to all subscribers of this uri."""
        key = str(uri)
        subs = list(self._resource_subscriptions.get(key, ()))  # copy to avoid mutation during iteration
        for sess in subs:
            try:
                await sess.send_resource_updated(key)
            except Exception:
                # stale/closed session; best-effort cleanup
                try:
                    self._resource_subscriptions[key].discard(sess)
                except Exception:
                    pass

    def _register_builtin_run_resources(self) -> None:
        """Register built-in resources for run inputs/outputs/conversation & span payloads."""

        async def _find_run_span_id(ctx: "Context", run_id: str) -> str | None:  # type: ignore[name-defined]
            events = await ctx.wrp.store.load_telemetry(run_id)
            for ev in events:
                # look for the single run-start span
                if (
                    getattr(ev, "kind", None) == "span"
                    and getattr(ev, "span_kind", None) == "run"
                    and getattr(ev, "phase", None) == "start"
                ):
                    return getattr(ev, "span_id", None)
            return None

        @self.resource("resource://runs/{run_id}/input", mime_type="application/json", title="Run Input")
        async def _run_input(run_id: str, ctx: "Context") -> Any:  # type: ignore[name-defined]
            span_id = await _find_run_span_id(ctx, run_id)
            if not span_id:
                return None
            env = await ctx.wrp.store.get_span_payload(run_id, span_id)
            if not env:
                return None
            env_dict = env.model_dump()
            sanitized = sanitize_envelope_dict(env_dict, ctx.wrp.telemetry_policy)
            part = sanitized.get("capture", {}).get("start")
            return part.get("data") if part else None

        @self.resource("resource://runs/{run_id}/output", mime_type="application/json", title="Run Output")
        async def _run_output(run_id: str, ctx: "Context") -> Any:  # type: ignore[name-defined]
            span_id = await _find_run_span_id(ctx, run_id)
            if not span_id:
                return None
            env = await ctx.wrp.store.get_span_payload(run_id, span_id)
            if not env:
                return None
            env_dict = env.model_dump()
            sanitized = sanitize_envelope_dict(env_dict, ctx.wrp.telemetry_policy)
            part = sanitized.get("capture", {}).get("end")
            return part.get("data") if part else None

        @self.resource("resource://runs/{run_id}/conversations", mime_type="application/json", title="Run Conversations (Index)")
        async def _run_conversations_index(run_id: str, ctx: "Context") -> Any:  # type: ignore[name-defined]
            items = await ctx.wrp.store.load_conversation(run_id)
            channels = sorted({getattr(it, "channel", "default") for it in items})
            return {"channels": channels}

        @self.resource(
            "resource://runs/{run_id}/conversations/{channel}",
            mime_type="application/json",
            title="Run Conversations (Channel)",
        )
        async def _run_conversation_channel(run_id: str, channel: str, ctx: "Context") -> Any:  # type: ignore[name-defined]
            all_items = await ctx.wrp.store.load_conversation(run_id)
            filtered = [it for it in all_items if it.channel == channel]
            sanitized = sanitize_conversation_items(filtered, ctx.wrp.conversation_policy)
            return sanitized

        @self.resource(
            "resource://runs/{run_id}/telemetry/spans/{span_id}/payload",
            mime_type="application/json",
            title="Span Payload",
        )
        async def _span_payload(run_id: str, span_id: str, ctx: "Context") -> Any:  # type: ignore[name-defined]
            env = await ctx.wrp.store.get_span_payload(run_id, span_id)
            if not env:
                return None
            env_dict = env.model_dump()
            return sanitize_envelope_dict(env_dict, ctx.wrp.telemetry_policy)

        # ---- Spans (derived from plaintext telemetry events) ---------------
        @self.resource(
            "resource://runs/{run_id}/telemetry/spans",
            mime_type="application/json",
            title="Span Index",
        )
        async def _spans(run_id: str, ctx: "Context") -> Any:  # type: ignore[name-defined]
            # Load only span-kind events for this run, ascending by ts
            events = await ctx.wrp.store.load_telemetry(run_id, kinds={"span"})
            spans = build_span_index(events)
            return serialize_span_list(spans)

        @self.resource(
            "resource://runs/{run_id}/telemetry/spans/{span_id}",
            mime_type="application/json",
            title="Span Detail",
        )
        async def _span_detail(run_id: str, span_id: str, ctx: "Context") -> Any:  # type: ignore[name-defined]
            events = await ctx.wrp.store.load_telemetry(run_id, kinds={"span"})
            spans = build_span_index(events)
            return serialize_span_detail(spans, span_id)

    def _register_builtin_workflow_settings_endpoints(self) -> None:
        """Expose workflow settings read/schema as resources and a simple HTTP mutation endpoint."""
        # MCP resources (read-only)
        @self.resource(
            "resource://workflows/{workflow}/settings",
            mime_type="application/json",
            title="Workflow Settings",
        )
        async def _wf_settings_resource(workflow: str, ctx: "Context") -> Any:  # type: ignore[name-defined]
            await self._workflow_manager.load_persisted_settings_if_needed(workflow)
            cur = self._workflow_manager.get_settings(workflow)
            if cur is None:
                return None
            wf = self._workflow_manager.get(workflow)
            return {
                "values": cur.model_dump(),
                "overridden": self._workflow_manager.settings_overridden(workflow),
                "allowOverride": bool(wf.settings_allow_override) if wf else False,
                "locked": list(getattr(cur.__class__, "locked", set())),
            }

        @self.resource(
            "resource://workflows/{workflow}/settings/schema",
            mime_type="application/json",
            title="Workflow Settings Schema",
        )
        async def _wf_settings_schema_resource(workflow: str, ctx: "Context") -> Any:  # type: ignore[name-defined]
            wf = self._workflow_manager.get(workflow)
            if not wf or wf.settings_default is None:
                return None
            return wf.settings_default.__class__.model_json_schema(by_alias=True)

        # HTTP route for GET/PUT
        async def _wf_settings_http(request: Request) -> Response:
            workflow = request.path_params.get("workflow")
            if request.method == "GET":
                await self._workflow_manager.load_persisted_settings_if_needed(workflow)
                cur = self._workflow_manager.get_settings(workflow)
                if cur is None:
                    return JSONResponse(None)
                wf = self._workflow_manager.get(workflow)
                return JSONResponse({
                    "values": cur.model_dump(),
                    "overridden": self._workflow_manager.settings_overridden(workflow),
                    "allowOverride": bool(wf.settings_allow_override) if wf else False,
                    "locked": list(getattr(cur.__class__, "locked", set())),
                })
            # PUT -> merge (partial upsert)
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "Invalid JSON"}, status_code=400)
            try:
                inst = await self._workflow_manager.update_settings(workflow, body or {})
                wf = self._workflow_manager.get(workflow)
                return JSONResponse({
                    "values": inst.model_dump(),
                    "overridden": True,
                    "allowOverride": bool(wf.settings_allow_override) if wf else False,
                    "locked": list(getattr(inst.__class__, "locked", set())),
                })
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=400)

        self._custom_starlette_routes.append(
            Route("/workflows/{workflow}/settings", endpoint=_wf_settings_http, methods=["GET", "PUT"])
        )

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

    async def list_workflows(self) -> ListWorkflowsResult:
        """Return public descriptors for all registered workflows (no pagination)."""
        return ListWorkflowsResult(workflows=self._workflow_manager.list_descriptors())

    async def run_workflow(self, name: str, wf_input: dict[str, Any]) -> RunWorkflowResult:
        """Run a workflow by name with a dict payload matching its WorkflowInput.

        Note: This uses the current request context if available. Outside a request,
        the context is still constructed by `get_context()`, but most MCP-bound
        features (like logging or progress reporting) will raise an error if called.
        """
        context = self.get_context()
        # Ensure effective settings are hydrated from durable store before executing
        await self._workflow_manager.load_persisted_settings_if_needed(name)
        return await self._workflow_manager.run(name, wf_input, context)

    async def list_resources(self) -> list[MCPResource]:
        """List all available resources."""

        resources = self._resource_manager.list_resources()
        return [
            MCPResource(
                uri=resource.uri,
                name=resource.name or "",
                title=resource.title,
                description=resource.description,
                mimeType=resource.mime_type,
                icons=resource.icons,
            )
            for resource in resources
        ]

    async def list_resource_templates(self) -> list[MCPResourceTemplate]:
        templates = self._resource_manager.list_templates()
        return [
            MCPResourceTemplate(
                uriTemplate=template.uri_template,
                name=template.name,
                title=template.title,
                description=template.description,
                mimeType=template.mime_type,
                icons=template.icons,
            )
            for template in templates
        ]

    async def read_resource(self, uri: AnyUrl | str) -> Iterable[ReadResourceContents]:
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
        icons: list[Icon] | None = None,
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
        icons: list[Icon] | None = None,
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
        icons: list[Icon] | None = None,
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
        async with stdio_server() as (read_stream, write_stream):
            await self._wrp_server.run(
                read_stream,
                write_stream,
                self._wrp_server.create_initialization_options(),
            )

    async def run_sse_async(self, mount_path: str | None = None) -> None:
        """Run the server using SSE transport."""
        import uvicorn

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
        return await self._wrp.read_resource(uri)

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
        return cfg.copy(deep=True) if cfg is not None else None