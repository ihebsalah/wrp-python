# wrp/server/lowlevel/server.py
"""
WRP Server Module

This module provides a framework for creating an WRP (Workflow Runtime Protocol) server.
It allows you to easily define and handle various types of requests and notifications
in an asynchronous manner.

Usage:
1. Create a Server instance:
   server = Server("your_server_name")

2. Define request handlers using decorators:
    @server.list_workflows()
    async def handle_list_workflows(request: WRPListWorkflowsRequest) -> WRPListWorkflowsResult:
       # Implementation

    @server.run_workflow()
    async def handle_run_workflow(
       name: str, input: dict | None
    ) -> RunWorkflowResult:
       # Implementation

    @server.list_resource_templates()
    async def handle_list_resource_templates() -> list[types.ResourceTemplate]:
       # Implementation

3. Define notification handlers if needed:
   @server.progress_notification()
   async def handle_progress(
       progress_token: str | int, progress: float, total: float | None,
       message: str | None
   ) -> None:
       # Implementation

4. Run the server:
   async def main():
       async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
           await server.run(
               read_stream,
               write_stream,
               InitializationOptions(
                   server_name="your_server_name",
                   server_version="your_version",
                   capabilities=server.get_capabilities(
                       notification_options=NotificationOptions(),
                       experimental_capabilities={},
                   ),
               ),
           )

   asyncio.run(main())

The Server class provides methods to register handlers for various WRP requests and
notifications. It automatically manages the request context and handles incoming
messages from the client.
"""

from __future__ import annotations as _annotations

import contextvars
import logging
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from typing import Any, Generic, TypeAlias, cast

import anyio
import jsonschema
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic import AnyUrl
from typing_extensions import TypeVar

from mcp.server.lowlevel.func_inspection import create_call_wrapper
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.shared.message import ServerMessageMetadata, SessionMessage

import wrp.types as types
from wrp.shared.exceptions import WrpError
from wrp.shared.context import RequestContext
from wrp.shared.session import RequestResponder
from wrp.server.models import InitializationOptions
from wrp.server.session import ServerSession

logger = logging.getLogger(__name__)

LifespanResultT = TypeVar("LifespanResultT", default=Any)
RequestT = TypeVar("RequestT", default=Any)

# This will be properly typed in each Server instance's context
request_ctx: contextvars.ContextVar[RequestContext[ServerSession, Any, Any]] = contextvars.ContextVar("request_ctx")


class NotificationOptions:
    def __init__(
        self,
        resources_changed: bool = False,
        workflows_changed: bool = False,
    ):
        self.resources_changed = resources_changed
        self.workflows_changed = workflows_changed


@asynccontextmanager
async def lifespan(_: Server[LifespanResultT, RequestT]) -> AsyncIterator[dict[str, Any]]:
    """Default lifespan context manager that does nothing.

    Args:
        server: The server instance this lifespan is managing

    Returns:
        An empty context object
    """
    yield {}


class Server(Generic[LifespanResultT, RequestT]):
    def __init__(
        self,
        name: str,
        version: str | None = None,
        instructions: str | None = None,
        website_url: str | None = None,
        icons: list[types.Icon] | None = None,
        lifespan: Callable[
            [Server[LifespanResultT, RequestT]],
            AbstractAsyncContextManager[LifespanResultT],
        ] = lifespan,
    ):
        self.name = name
        self.version = version
        self.instructions = instructions
        self.website_url = website_url
        self.icons = icons
        self.lifespan = lifespan
        self.request_handlers: dict[type, Callable[..., Awaitable[types.ServerResult]]] = {
            types.PingRequest: _ping_handler,
        }
        self.notification_handlers: dict[type, Callable[..., Awaitable[None]]] = {}
        self._workflow_cache: dict[str, types.WorkflowDescriptor] = {}
        logger.debug("Initializing server %r", name)

    def create_initialization_options(
        self,
        notification_options: NotificationOptions | None = None,
        experimental_capabilities: dict[str, dict[str, Any]] | None = None,
    ) -> InitializationOptions:
        """Create initialization options from this server instance."""

        def pkg_version(package: str) -> str:
            try:
                from importlib.metadata import version

                return version(package)
            except Exception:
                pass

            return "unknown"

        return InitializationOptions(
            server_name=self.name,
            server_version=self.version if self.version else pkg_version("wrp"),
            capabilities=self.get_capabilities(
                notification_options or NotificationOptions(),
                experimental_capabilities or {},
            ),
            instructions=self.instructions,
            website_url=self.website_url,
            icons=self.icons,
        )

    def get_capabilities(
        self,
        notification_options: NotificationOptions,
        experimental_capabilities: dict[str, dict[str, Any]],
    ) -> types.ServerCapabilities:
        """Convert existing handlers to a ServerCapabilities object."""
        resources_capability = None
        logging_capability = None
        workflows_capability = None

        # Set resource capabilities if handler exists
        if types.ListResourcesRequest in self.request_handlers:
            resources_capability = types.ResourcesCapability(
                subscribe=True, listChanged=notification_options.resources_changed
            )

        # Set workflow capabilities if handler exists
        if types.ListWorkflowsRequest in self.request_handlers:
            workflows_capability = types.WorkflowsCapability(listChanged=notification_options.workflows_changed)

        # Set logging capabilities if handler exists
        if types.SetLevelRequest in self.request_handlers:
            logging_capability = types.LoggingCapability()

        return types.ServerCapabilities(
            resources=resources_capability,
            workflows=workflows_capability,
            logging=logging_capability,
            experimental=experimental_capabilities,
        )

    @property
    def request_context(
        self,
    ) -> RequestContext[ServerSession, LifespanResultT, RequestT]:
        """If called outside of a request context, this will raise a LookupError."""
        return request_ctx.get()

    def list_resources(self):
        def decorator(
            func: Callable[[], Awaitable[list[types.Resource]]]
            | Callable[[types.ListResourcesRequest], Awaitable[types.ListResourcesResult]],
        ):
            logger.debug("Registering handler for ListResourcesRequest")

            wrapper = create_call_wrapper(func, types.ListResourcesRequest)

            async def handler(req: types.ListResourcesRequest):
                result = await wrapper(req)
                # Handle both old style (list[Resource]) and new style (ListResourcesResult)
                if isinstance(result, types.ListResourcesResult):
                    return types.ServerResult(result)
                else:
                    # Old style returns list[Resource]
                    return types.ServerResult(types.ListResourcesResult(resources=result))

            self.request_handlers[types.ListResourcesRequest] = handler
            return func

        return decorator

    def list_resource_templates(self):
        def decorator(func: Callable[[], Awaitable[list[types.ResourceTemplate]]]):
            logger.debug("Registering handler for ListResourceTemplatesRequest")

            async def handler(_: Any):
                templates = await func()
                return types.ServerResult(types.ListResourceTemplatesResult(resourceTemplates=templates))

            self.request_handlers[types.ListResourceTemplatesRequest] = handler
            return func

        return decorator

    def read_resource(self):
        def decorator(
            func: Callable[[AnyUrl], Awaitable[str | bytes | Iterable[ReadResourceContents]]],
        ):
            logger.debug("Registering handler for ReadResourceRequest")

            async def handler(req: types.ReadResourceRequest):
                result = await func(req.params.uri)

                def create_content(data: str | bytes, mime_type: str | None):
                    match data:
                        case str() as data:
                            return types.TextResourceContents(
                                uri=req.params.uri,
                                text=data,
                                mimeType=mime_type or "text/plain",
                            )
                        case bytes() as data:
                            import base64

                            return types.BlobResourceContents(
                                uri=req.params.uri,
                                blob=base64.b64encode(data).decode(),
                                mimeType=mime_type or "application/octet-stream",
                            )

                match result:
                    case str() | bytes() as data:
                        warnings.warn(
                            "Returning str or bytes from read_resource is deprecated. "
                            "Use Iterable[ReadResourceContents] instead.",
                            DeprecationWarning,
                            stacklevel=2,
                        )
                        content = create_content(data, None)
                    case Iterable() as contents:
                        contents_list = [
                            create_content(content_item.content, content_item.mime_type) for content_item in contents
                        ]
                        return types.ServerResult(
                            types.ReadResourceResult(
                                contents=contents_list,
                            )
                        )
                    case _:
                        raise ValueError(f"Unexpected return type from read_resource: {type(result)}")

                return types.ServerResult(
                    types.ReadResourceResult(
                        contents=[content],
                    )
                )

            self.request_handlers[types.ReadResourceRequest] = handler
            return func

        return decorator

    def set_logging_level(self):
        def decorator(func: Callable[[types.LoggingLevel], Awaitable[None]]):
            logger.debug("Registering handler for SetLevelRequest")

            async def handler(req: types.SetLevelRequest):
                await func(req.params.level)
                return types.ServerResult(types.EmptyResult())

            self.request_handlers[types.SetLevelRequest] = handler
            return func

        return decorator

    def subscribe_resource(self):
        def decorator(func: Callable[[AnyUrl], Awaitable[None]]):
            logger.debug("Registering handler for SubscribeRequest")

            async def handler(req: types.SubscribeRequest):
                await func(req.params.uri)
                return types.ServerResult(types.EmptyResult())

            self.request_handlers[types.SubscribeRequest] = handler
            return func

        return decorator

    def unsubscribe_resource(self):
        def decorator(func: Callable[[AnyUrl], Awaitable[None]]):
            logger.debug("Registering handler for UnsubscribeRequest")

            async def handler(req: types.UnsubscribeRequest):
                await func(req.params.uri)
                return types.ServerResult(types.EmptyResult())

            self.request_handlers[types.UnsubscribeRequest] = handler
            return func

        return decorator

    def list_workflows(self):
        def decorator(
            func: Callable[[types.ListWorkflowsRequest], Awaitable[types.ListWorkflowsResult]],
        ):
            logger.debug("Registering handler for ListWorkflowsRequest")

            wrapper = create_call_wrapper(func, types.ListWorkflowsRequest)

            async def handler(req: types.ListWorkflowsRequest):
                # Always expect a WRPListWorkflowsResult
                result = await wrapper(req)

                # Refresh cache from canonical result
                self._workflow_cache.clear()
                for wf in result.workflows:
                    self._workflow_cache[wf.name] = wf

                # Return the canonical result as-is
                return types.ServerResult(result)

            # Register handler keyed by our WRP request type
            self.request_handlers[types.ListWorkflowsRequest] = handler
            return func

        return decorator

    async def _get_cached_workflow_definition(self, workflow_name: str) -> types.WorkflowDescriptor | None:
        """Get workflow definition from cache, refreshing if necessary."""
        if workflow_name not in self._workflow_cache:
            if types.ListWorkflowsRequest in self.request_handlers:
                logger.debug("Workflow cache miss for %s, refreshing cache", workflow_name)
                # Trigger a refresh — handler supports being called with None
                await self.request_handlers[types.ListWorkflowsRequest](types.ListWorkflowsRequest())

        wf = self._workflow_cache.get(workflow_name)
        if wf is None:
            logger.warning("Workflow '%s' not listed, no validation will be performed", workflow_name)
        return wf

    def run_workflow(self, *, validate_input: bool = True):
        """Register a workflow run handler.

        The handler validates input against the workflow's inputSchema (if available),
        calls the provided function, and wraps the returned result in RunWorkflowResult.
        """

        def decorator(
            func: Callable[[str, dict[str, Any]], Awaitable[types.RunWorkflowResult]]
            | Callable[[types.RunWorkflowRequest], Awaitable[types.RunWorkflowResult]],
        ):
            logger.debug("Registering handler for RunWorkflowRequest")

            async def handler(req: types.RunWorkflowRequest):
                try:
                    wf_name = req.params.name
                    wf_input = req.params.input or {}

                    # Input validation (if we have a descriptor cached)
                    if validate_input:
                        wf = await self._get_cached_workflow_definition(wf_name)
                        if wf and getattr(wf, "inputSchema", None):
                            try:
                                jsonschema.validate(instance=wf_input, schema=wf.inputSchema)
                            except jsonschema.ValidationError as e:
                                return types.ServerResult(
                                    types.RunWorkflowResult(isError=True, error=f"Input validation error: {e.message}")
                                )

                    # Call function — support both signatures
                    if hasattr(func, "__call__"):
                        try:
                            # Prefer (name, input) signature
                            result = await func(wf_name, wf_input)  # type: ignore[arg-type]
                        except TypeError:
                            # Fallback to request model signature
                            result = await func(req)  # type: ignore[arg-type]
                    else:
                        return types.ServerResult(types.RunWorkflowResult(isError=True, error="Invalid workflow handler"))

                    # Defensive normalization: ensure result is a RunWorkflowResult
                    if not isinstance(result, types.RunWorkflowResult):
                        # Try to coerce from a dict with matching fields
                        if isinstance(result, dict):
                            result = types.RunWorkflowResult(**result)
                        else:
                            return types.ServerResult(
                                types.RunWorkflowResult(
                                    isError=True,
                                    error=f"Unexpected return type from workflow: {type(result).__name__}",
                                )
                            )

                    # Output validation (if we have a descriptor & non-error output)
                    if not result.isError:
                        wf = await self._get_cached_workflow_definition(wf_name)
                        if wf and getattr(wf, "outputSchema", None) and result.output is not None:
                            try:
                                jsonschema.validate(instance=result.output, schema=wf.outputSchema)
                            except jsonschema.ValidationError as e:
                                return types.ServerResult(
                                    types.RunWorkflowResult(isError=True, error=f"Output validation error: {e.message}")
                                )

                    return types.ServerResult(result)
                except Exception as e:
                    logger.exception("Error running workflow %s", getattr(req.params, "name", "<unknown>"))
                    return types.ServerResult(types.RunWorkflowResult(isError=True, error=str(e)))

            self.request_handlers[types.RunWorkflowRequest] = handler
            return func

        return decorator

    def progress_notification(self):
        def decorator(
            func: Callable[[str | int, float, float | None, str | None], Awaitable[None]],
        ):
            logger.debug("Registering handler for ProgressNotification")

            async def handler(req: types.ProgressNotification):
                await func(
                    req.params.progressToken,
                    req.params.progress,
                    req.params.total,
                    req.params.message,
                )

            self.notification_handlers[types.ProgressNotification] = handler
            return func

        return decorator

    async def run(
        self,
        read_stream: MemoryObjectReceiveStream[SessionMessage | Exception],
        write_stream: MemoryObjectSendStream[SessionMessage],
        initialization_options: InitializationOptions,
        # When False, exceptions are returned as messages to the client.
        # When True, exceptions are raised, which will cause the server to shut down
        # but also make tracing exceptions much easier during testing and when using
        # in-process servers.
        raise_exceptions: bool = False,
        # When True, the server is stateless and
        # clients can perform initialization with any node. The client must still follow
        # the initialization lifecycle, but can do so with any available node
        # rather than requiring initialization for each connection.
        stateless: bool = False,
    ):
        async with AsyncExitStack() as stack:
            lifespan_context = await stack.enter_async_context(self.lifespan(self))
            session = await stack.enter_async_context(
                ServerSession(
                    read_stream,
                    write_stream,
                    initialization_options,
                    stateless=stateless,
                )
            )

            async with anyio.create_task_group() as tg:
                async for message in session.incoming_messages:
                    logger.debug("Received message: %s", message)

                    tg.start_soon(
                        self._handle_message,
                        message,
                        session,
                        lifespan_context,
                        raise_exceptions,
                    )

    async def _handle_message(
        self,
        message: RequestResponder[types.ClientRequest, types.ServerResult] | types.ClientNotification | Exception,
        session: ServerSession,
        lifespan_context: LifespanResultT,
        raise_exceptions: bool = False,
    ):
        with warnings.catch_warnings(record=True) as w:
            # TODO(Marcelo): We should be checking if message is Exception here.
            match message:  # type: ignore[reportMatchNotExhaustive]
                case RequestResponder(request=types.ClientRequest(root=req)) as responder:
                    with responder:
                        await self._handle_request(message, req, session, lifespan_context, raise_exceptions)
                case types.ClientNotification(root=notify):
                    await self._handle_notification(notify)

            for warning in w:
                logger.info("Warning: %s: %s", warning.category.__name__, warning.message)

    async def _handle_request(
        self,
        message: RequestResponder[types.ClientRequest, types.ServerResult],
        req: Any,
        session: ServerSession,
        lifespan_context: LifespanResultT,
        raise_exceptions: bool,
    ):
        logger.info("Processing request of type %s", type(req).__name__)
        if handler := self.request_handlers.get(type(req)):  # type: ignore
            logger.debug("Dispatching request of type %s", type(req).__name__)

            token = None
            try:
                # Extract request context from message metadata
                request_data = None
                if message.message_metadata is not None and isinstance(message.message_metadata, ServerMessageMetadata):
                    request_data = message.message_metadata.request_context

                # Set our global state that can be retrieved via
                # app.get_request_context()
                token = request_ctx.set(
                    RequestContext(
                        message.request_id,
                        message.request_meta,
                        session,
                        lifespan_context,
                        request=request_data,
                    )
                )
                response = await handler(req)
            except WrpError as err:
                response = err.error
            except anyio.get_cancelled_exc_class():
                logger.info(
                    "Request %s cancelled - duplicate response suppressed",
                    message.request_id,
                )
                return
            except Exception as err:
                if raise_exceptions:
                    raise err
                response = types.ErrorData(code=0, message=str(err), data=None)
            finally:
                # Reset the global state after we are done
                if token is not None:
                    request_ctx.reset(token)

            await message.respond(response)
        else:
            await message.respond(
                types.ErrorData(
                    code=types.METHOD_NOT_FOUND,
                    message="Method not found",
                )
            )

        logger.debug("Response sent")

    async def _handle_notification(self, notify: Any):
        if handler := self.notification_handlers.get(type(notify)):  # type: ignore
            logger.debug("Dispatching notification of type %s", type(notify).__name__)

            try:
                await handler(notify)
            except Exception:
                logger.exception("Uncaught exception in notification handler")


async def _ping_handler(request: types.PingRequest) -> types.ServerResult:
    return types.ServerResult(types.EmptyResult())