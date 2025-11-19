# tests/unit/server/test_routing.py
import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock
import wrp.types as types
from pydantic import AnyUrl
from mcp.server.lowlevel.helper_types import ReadResourceContents
from wrp.server.lowlevel.server import Server, NotificationOptions
from wrp.shared.session import RequestResponder
from wrp.types import (
    ClientRequest,
    EmptyResult,
    ErrorData,
    ListResourcesRequest,
    ListResourcesResult,
    METHOD_NOT_FOUND,
    PingRequest,
    ProgressNotification,
    ProgressNotificationParams,
    ReadResourceRequest,
    ReadResourceRequestParams,
    ReadResourceResult,
    RunWorkflowRequest,
    RunWorkflowRequestParams,
    RunWorkflowResult,
    ServerResult,
    WorkflowDescriptor,
)


@pytest.mark.anyio
async def test_server_workflow_registration():
    """
    Verifies that the Server class correctly registers workflow handlers
    and routing capabilities.
    """
    server = Server("test-router")

    # 1. Define and register a dummy workflow
    # We return RunWorkflowResult explicitly to satisfy the type checker,
    # though your runtime supports dicts too.
    @server.run_workflow(validate_input=False)
    async def _run_workflow(name: str, wf_input: dict[str, Any]) -> types.RunWorkflowResult:
        return RunWorkflowResult(
            output={"status": "success", "echo": wf_input.get("value")}
        )

    # 1b. Register the list_workflows handler.
    # This is required for the server to announce workflow capabilities.
    @server.list_workflows()
    async def _list_workflows(_req: types.ListWorkflowsRequest,) -> types.ListWorkflowsResult:
        return types.ListWorkflowsResult(workflows=[])

    # 2. Check Capabilities generation
    caps = server.get_capabilities(
        notification_options=NotificationOptions(),
        experimental_capabilities={}
    )
    assert caps.workflows is not None

    # 3. Verify internal handler registry
    assert RunWorkflowRequest in server.request_handlers

    # 4. Simulate a direct handler call
    handler = server.request_handlers[RunWorkflowRequest]

    req = RunWorkflowRequest(
        params=RunWorkflowRequestParams(
            name="test-flow",
            input={"value": "hello"}
        )
    )

    response = await handler(req)

    # 5. Type Narrowing: Assert the type before accessing attributes
    # response is ServerResult, response.root is the Union
    result = response.root

    # This assertion tells Pylance: "If it's not RunWorkflowResult, fail the test here"
    assert isinstance(result, RunWorkflowResult)

    # Now Pylance knows 'result' has .output and .isError
    assert result.output == {"status": "success", "echo": "hello"}
    assert result.isError is False


@pytest.mark.anyio
async def test_server_resource_routing():
    """
    Verifies that the Server correctly registers resource handlers
    and wraps return values into WRP protocol objects.
    """
    server = Server("test-resources")

    # 1. Register a simple resource
    # The decorator logic in `lowlevel/server.py` wraps the return value
    # into a ReadResourceResult automatically.
    # NOTE: lowlevel.Server uses read_resource(), not resource() (which is high-level WRP)
    @server.read_resource()
    async def _get_resource(uri: AnyUrl) -> list[ReadResourceContents]:
        return [ReadResourceContents(content="resource content", mime_type="text/plain")]

    @server.list_resources()
    async def _list_resources() -> list[types.Resource]:
        return [types.Resource(uri=AnyUrl("resource://test"), name="test")]

    # 2. Verify Capabilities
    caps = server.get_capabilities(NotificationOptions(), {})
    assert caps.resources is not None

    # 3. Test ReadResource Routing
    handler = server.request_handlers[ReadResourceRequest]
    req = ReadResourceRequest(params=ReadResourceRequestParams(uri=AnyUrl("resource://test")))

    response = await handler(req)
    result = response.root

    assert isinstance(result, ReadResourceResult)
    assert len(result.contents) == 1
    content = result.contents[0]
    assert isinstance(content, types.TextResourceContents)
    assert content.text == "resource content"
    assert content.mimeType == "text/plain"


@pytest.mark.anyio
async def test_server_handler_exception_handling():
    """
    Verifies that if a workflow handler raises an exception, the Server
    catches it and returns an error result, rather than crashing.
    """
    server = Server("test-errors")

    @server.run_workflow(validate_input=False)
    async def _crashing_workflow(name: str, wf_input: dict) -> types.RunWorkflowResult:
        raise ValueError("Boom!")

    handler = server.request_handlers[RunWorkflowRequest]
    req = RunWorkflowRequest(
        params=RunWorkflowRequestParams(name="crash", input={})
    )

    response = await handler(req)
    result = response.root

    assert isinstance(result, RunWorkflowResult)
    assert result.isError is True
    assert "Boom!" in str(result.error)


@pytest.mark.anyio
async def test_server_capabilities_logic():
    """
    Verifies that the Server correctly toggles capability flags based on
    which handlers are registered. This ensures the 'initialize' response
    will be correct.
    """
    server = Server("test-caps")

    # 1. Initially, no capabilities
    caps = server.get_capabilities(NotificationOptions(), {})
    assert caps.workflows is None
    assert caps.resources is None

    # 2. Register Workflows -> workflows capability appears
    @server.list_workflows()
    async def _list_wf(_): return types.ListWorkflowsResult(workflows=[])

    caps = server.get_capabilities(NotificationOptions(), {})
    assert caps.workflows is not None
    assert caps.workflows.settings is None  # Settings not registered yet

    # 3. Register Workflow Settings -> workflows.settings.read becomes True
    @server.workflow_settings_read()
    async def _read_settings(name): return types.WorkflowSettingsReadResult()

    caps = server.get_capabilities(NotificationOptions(), {})
    assert caps.workflows is not None
    assert caps.workflows.settings is not None
    assert caps.workflows.settings.read is True
    assert caps.workflows.settings.update is None  # We didn't register update

    # 4. Register Logging -> logging capability appears
    @server.set_logging_level()
    async def _set_log(level): pass

    caps = server.get_capabilities(NotificationOptions(), {})
    assert caps.logging is not None


@pytest.mark.anyio
async def test_server_notification_registration():
    """
    Verifies that notification handlers (which don't return results)
    are registered in the correct registry.
    """
    server = Server("test-notif")

    @server.progress_notification()
    async def _on_progress(token, p, t, m):
        pass

    # Notifications go into a different dict than requests
    assert ProgressNotification in server.notification_handlers
    # Ensure the decorator didn't touch request handlers
    assert PingRequest in server.request_handlers


@pytest.mark.anyio
async def test_server_ping_default():
    """
    Verifies that a new Server instance handles PingRequest by default.
    """
    server = Server("test-ping")

    # Ping should be registered by __init__
    assert PingRequest in server.request_handlers

    handler = server.request_handlers[PingRequest]
    response = await handler(PingRequest())

    assert isinstance(response.root, EmptyResult)


@pytest.mark.anyio
async def test_server_context_injection():
    """
    Verifies that the Server injects the RequestContext into the
    contextvar during request handling, allowing handlers to access it.
    """
    server = Server("test-ctx")

    # 1. Define a handler that checks if request_context is set
    @server.run_workflow(validate_input=False)
    async def _context_aware_workflow(name: str, input: dict) -> types.RunWorkflowResult:
        # This will raise LookupError if context is not set
        ctx = server.request_context
        return types.RunWorkflowResult(output={"session_id": ctx.request_id})

    # 2. Mock the machinery required to call _handle_request
    # We can't just call handler() directly because _handle_request is what sets the context

    mock_session = MagicMock()
    mock_responder = AsyncMock(spec=RequestResponder)
    mock_responder.request_id = "req-123"
    mock_responder.request_meta = None
    mock_responder.session = mock_session
    mock_responder.message_metadata = None

    # 3. Invoke _handle_request directly (private method, but necessary for unit testing logic)
    # This simulates the server receiving a message
    req = RunWorkflowRequest(params=RunWorkflowRequestParams(name="ctx-flow", input={}))

    # We mock the respond method to capture the result
    async def capture_response(response: ServerResult | ErrorData):
        assert isinstance(response, ServerResult)
        result = response.root
        assert isinstance(result, RunWorkflowResult)
        assert result.output is not None
        assert result.output["session_id"] == "req-123"

    mock_responder.respond = capture_response

    # 4. Run it
    # We pass None for lifespan_context as we aren't testing lifespan here
    await server._handle_request(
        message=mock_responder,
        req=req,
        session=mock_session,
        lifespan_context={},
        raise_exceptions=True
    )


@pytest.mark.anyio
async def test_server_input_validation():
    """
    Verifies that the Server enforces JSON schema validation when
    validate_input=True.
    """
    server = Server("test-validation")

    # 1. Register list_workflows to provide the schema
    @server.list_workflows()
    async def _list_workflows(_req: types.ListWorkflowsRequest,) -> types.ListWorkflowsResult:
        return types.ListWorkflowsResult(workflows=[
            WorkflowDescriptor(
                name="strict-flow",
                inputSchema={"type": "object", "properties": {"age": {"type": "integer"}}, "required": ["age"]},
                outputSchema={}
            )
        ])

    # 2. Register the workflow with validation enabled
    @server.run_workflow(validate_input=True)
    async def _run(name: str, input: dict):
        return types.RunWorkflowResult(output={})

    handler = server.request_handlers[RunWorkflowRequest]

    # 3. Send INVALID input (string instead of integer)
    req = RunWorkflowRequest(
        params=RunWorkflowRequestParams(name="strict-flow", input={"age": "not-a-number"})
    )

    response = await handler(req)
    result = response.root

    # 4. Assert validation failed gracefully
    assert isinstance(result, RunWorkflowResult)
    assert result.isError is True
    assert "Input validation error" in str(result.error)


@pytest.mark.anyio
async def test_server_method_not_found():
    """
    Verifies that the Server returns METHOD_NOT_FOUND for unknown request types.
    """
    server = Server("test-404")

    # Mock the responder machinery
    mock_responder = AsyncMock(spec=RequestResponder)
    mock_responder.request_id = 1

    # Create a request type that isn't registered
    class UnknownRequest(ClientRequest):
        pass

    # We mock respond to capture the error
    async def capture_response(response):
        assert isinstance(response, ErrorData)
        assert response.code == METHOD_NOT_FOUND
        assert response.message == "Method not found"

    mock_responder.respond = capture_response

    # Invoke _handle_request with an unknown request type
    # We pass a dummy object that isn't in server.request_handlers
    await server._handle_request(
        message=mock_responder,
        req="some-unknown-object",
        session=MagicMock(),
        lifespan_context={},
        raise_exceptions=False
    )


@pytest.mark.anyio
async def test_server_notification_dispatch():
    """
    Verifies that _handle_notification correctly locates and awaits
    the registered handler, and that the decorator unpacks arguments.
    """
    server = Server("test-notif-exec")

    # 1. Create a mock to capture the unpacked arguments
    mock_impl = AsyncMock()

    # 2. Register via decorator to test the full chain (dispatch + unpacking)
    @server.progress_notification()
    async def _on_progress(token, progress, total, message):
        await mock_impl(token, progress, total, message)

    # 3. Create the notification object
    notif = ProgressNotification(
        params=ProgressNotificationParams(progressToken="pt", progress=0.5)
    )

    # 4. Invoke the private handler method
    await server._handle_notification(notif)

    # 5. Assert the implementation was called with unpacked params
    mock_impl.assert_awaited_once()
    call_args = mock_impl.call_args
    assert call_args[0][0] == "pt"  # progressToken
    assert call_args[0][1] == 0.5   # progress