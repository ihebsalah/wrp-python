# tests/unit/server/test_server_session.py
import pytest
import anyio
from pydantic import AnyUrl, FileUrl
from wrp.server.session import ServerSession, InitializationState
from wrp.server.models import InitializationOptions
from wrp.types import (
    ServerCapabilities,
    ClientCapabilities,
    InitializeRequestParams,
    Implementation,
    JSONRPCRequest,
    JSONRPCNotification,
    JSONRPCResponse,
    JSONRPCMessage,
    InitializeResult,
    ElicitResult,
    RootsCapability,
    PingRequest,
    ProgressNotification,
    EmptyResult,
    ListRootsResult,
    Root,
    RunWorkflowRequestParams,
    JSONRPCError,
    INVALID_PARAMS,
    RunWorkflowRequest,
)
from wrp.shared.message import SessionMessage
from wrp.shared.session import RequestResponder

@pytest.fixture
def server_session_pair(memory_channel_pair):
    """
    Returns a ServerSession instance and the client-side streams to interact with it.
    """
    (client_read, client_write), (server_read, server_write) = memory_channel_pair
    
    init_options = InitializationOptions(
        server_name="unit-test-server",
        server_version="0.1.0",
        capabilities=ServerCapabilities()
    )
    
    session = ServerSession(
        read_stream=server_read,
        write_stream=server_write,
        init_options=init_options,
    )
    
    return session, client_read, client_write

@pytest.mark.anyio
async def test_server_session_initialization_flow(server_session_pair):
    """
    Verifies the strict initialization state machine of ServerSession:
    NotInitialized -> (Receive Initialize) -> Initializing -> (Receive Initialized) -> Initialized
    """
    session, client_read, client_write = server_session_pair
    
    async with session, client_read, client_write:
        # 1. Initial state check
        assert session._initialization_state == InitializationState.NotInitialized

        # 2. Send Initialize Request (Client -> Server)
        req = JSONRPCRequest(
            jsonrpc="2.0",
            id=1,
            method="initialize",
            params=InitializeRequestParams(
                protocolVersion="2025-11-01",
                capabilities=ClientCapabilities(roots=None),
                clientInfo=Implementation(name="test-client", version="1.0")
            ).model_dump(mode="json", by_alias=True)
        )
        await client_write.send(SessionMessage(JSONRPCMessage(req)))

        # 3. Receive Initialize Result (Server -> Client)
        response_msg = await client_read.receive()
        resp = response_msg.message.root
        assert isinstance(resp, JSONRPCResponse)
        result = InitializeResult.model_validate(resp.result)
        assert result.serverInfo.name == "unit-test-server"
        
        # State should be Initializing (waiting for initialized notification)
        assert session._initialization_state == InitializationState.Initializing

        # 4. Send Initialized Notification (Client -> Server)
        notif = JSONRPCNotification(
            jsonrpc="2.0",
            method="notifications/initialized",
            params={}
        )
        await client_write.send(SessionMessage(JSONRPCMessage(notif)))
        
        # Allow loop to process the notification
        await anyio.sleep(0.01)
        
        # State should be Initialized
        assert session._initialization_state == InitializationState.Initialized

@pytest.mark.anyio
async def test_server_session_check_capabilities(server_session_pair):
    """
    Verifies that check_client_capability correctly logic-checks the
    capabilities provided by the client during initialization.
    """
    session, _, client_write = server_session_pair
    
    async with session, client_write:
        # Manually simulate initialization state to populate _client_params
        # (Bypassing the full handshake for a focused unit test)
        session._client_params = InitializeRequestParams(
            protocolVersion="2025-11-01",
            capabilities=ClientCapabilities(
                roots=None,
                experimental={"test_feature": {"enabled": True}}
            ),
            clientInfo=Implementation(name="test", version="1")
        )
        
        # Case 1: Client missing capability entirely
        has_roots = session.check_client_capability(ClientCapabilities(roots=RootsCapability(listChanged=True)))
        assert has_roots is False 

        # Case 2: Client has exact match
        has_exp = session.check_client_capability(ClientCapabilities(experimental={"test_feature": {"enabled": True}}))
        assert has_exp is True

        # Case 3: Client has capability but value mismatch
        has_exp_mismatch = session.check_client_capability(ClientCapabilities(experimental={"test_feature": {"enabled": False}}))
        assert has_exp_mismatch is False

@pytest.mark.anyio
async def test_server_session_send_notification(server_session_pair):
    """
    Verifies that server-initiated notifications (like logs) are correctly
    serialized and sent to the client stream.
    """
    session, client_read, _ = server_session_pair
    
    async with session, client_read:
        await session.send_log_message(level="info", data="test log", logger="test")
        
        msg = await client_read.receive()
        root = msg.message.root
        assert isinstance(root, JSONRPCNotification)
        assert root.method == "notifications/message"
        assert root.params is not None
        assert root.params["level"] == "info"
        assert root.params["data"] == "test log"

@pytest.mark.anyio
async def test_server_session_elicit_request(server_session_pair):
    """
    Verifies that the server can send a Request (elicit) to the client
    and await the response.
    """
    session, client_read, client_write = server_session_pair
    
    async with session, client_read, client_write:
        # Start a background task to act as the Client responding to the request
        async def client_responder():
            msg = await client_read.receive()
            req = msg.message.root
            assert isinstance(req, JSONRPCRequest)
            assert req.method == "elicitation/create"
            
            # Send response back to server
            resp = JSONRPCResponse(
                jsonrpc="2.0",
                id=req.id,
                result=ElicitResult(action="accept", content={"answer": "yes"}).model_dump(mode="json")
            )
            await client_write.send(SessionMessage(JSONRPCMessage(resp)))

        async with anyio.create_task_group() as tg:
            tg.start_soon(client_responder)
            
            # Server sends request and awaits response
            result = await session.elicit(
                message="Do you agree?",
                requestedSchema={"type": "object"}
            )
            
            assert result.action == "accept"
            assert result.content == {"answer": "yes"}

@pytest.mark.anyio
async def test_server_session_ping(server_session_pair):
    """
    Verifies server-initiated Ping.
    """
    session, client_read, client_write = server_session_pair
    
    async with session, client_read, client_write:
        async def client_responder():
            msg = await client_read.receive()
            req = msg.message.root
            assert isinstance(req, JSONRPCRequest)
            assert req.method == "ping"
            
            # Respond
            resp = JSONRPCResponse(
                jsonrpc="2.0",
                id=req.id,
                result=EmptyResult().model_dump(mode="json")
            )
            await client_write.send(SessionMessage(JSONRPCMessage(resp)))

        async with anyio.create_task_group() as tg:
            tg.start_soon(client_responder)
            await session.send_ping()

@pytest.mark.anyio
async def test_server_session_progress(server_session_pair):
    """
    Verifies server-initiated Progress notifications map arguments correctly.
    """
    session, client_read, _ = server_session_pair
    
    async with session, client_read:
        await session.send_progress_notification(
            progress_token="tok-1",
            progress=50.0,
            total=100.0,
            message="Halfway there"
        )
        
        msg = await client_read.receive()
        root = msg.message.root
        assert isinstance(root, JSONRPCNotification)
        assert root.method == "notifications/progress"
        assert root.params is not None
        assert root.params["progressToken"] == "tok-1"
        assert root.params["progress"] == 50.0
        assert root.params["message"] == "Halfway there"

@pytest.mark.anyio
async def test_server_session_enforces_initialization_order(server_session_pair):
    """
    Verifies that the server rejects requests sent BEFORE the initialization handshake is complete.
    """
    session, client_read, client_write = server_session_pair
    
    async with session, client_read, client_write:
        # 1. Send a random request (e.g., run workflow) BEFORE initialize
        req = JSONRPCRequest(
            jsonrpc="2.0",
            id=99,
            method="workflows/run",
            params=RunWorkflowRequestParams(name="foo").model_dump(mode="json")
        )
        await client_write.send(SessionMessage(JSONRPCMessage(req)))

        # 2. Server should respond with a JSON-RPC Error (caught in BaseSession loop)
        response_msg = await client_read.receive()
        resp = response_msg.message.root
        
        # Depending on implementation, this might be INVALID_PARAMS (validation fail) 
        # or a generic error because _received_request raised RuntimeError.
        # BaseSession catches exceptions in _received_request and sends JSONRPCError.
        assert isinstance(resp, JSONRPCError)
        # The ID must match
        assert resp.id == 99

@pytest.mark.anyio
async def test_server_session_list_roots(server_session_pair):
    """
    Verifies the server can request 'roots/list' from the client.
    """
    session, client_read, client_write = server_session_pair
    
    async with session, client_read, client_write:
        async def client_responder():
            msg = await client_read.receive()
            req = msg.message.root
            assert req.method == "roots/list"
            
            # Respond with a list of roots
            result = ListRootsResult(roots=[
                Root(uri=FileUrl("file:///tmp"), name="temp")
            ])
            
            resp = JSONRPCResponse(
                jsonrpc="2.0",
                id=req.id,
                result=result.model_dump(mode="json")
            )
            await client_write.send(SessionMessage(JSONRPCMessage(resp)))

        async with anyio.create_task_group() as tg:
            tg.start_soon(client_responder)
            
            result = await session.list_roots()
            assert len(result.roots) == 1
            assert str(result.roots[0].uri) == "file:///tmp"

@pytest.mark.anyio
async def test_server_session_specific_notifications(server_session_pair):
    """
    Verifies the helper methods for specific notifications map to the correct
    WRP protocol method strings.
    """
    session, client_read, _ = server_session_pair
    
    async with session, client_read:
        # 1. Resource Updated
        await session.send_resource_updated(uri=AnyUrl("resource://test"))
        msg = await client_read.receive()
        assert msg.message.root.method == "notifications/resources/updated"
        assert msg.message.root.params["uri"] == "resource://test"

        # 2. Resource List Changed
        await session.send_resource_list_changed()
        msg = await client_read.receive()
        assert msg.message.root.method == "notifications/resources/list_changed"

        # 3. Workflows List Changed
        await session.send_workflows_list_changed()
        msg = await client_read.receive()
        assert msg.message.root.method == "notifications/workflows/list_changed"

        # 4. System Events Updated (Complex payload)
        await session.send_system_events_updated(
            topic="runs/input",
            sequence=1,
            change="created",
            runs={"system_session_id": "s1", "run_id": "r1"}
        )
        msg = await client_read.receive()
        root = msg.message.root
        assert root.method == "notifications/systemEvents/updated"
        assert root.params["topic"] == "runs/input"
        assert root.params["runs"]["run_id"] == "r1"

@pytest.mark.anyio
async def test_server_session_stateless_mode(memory_channel_pair):
    """
    Verifies that if stateless=True, the session starts as Initialized
    and accepts requests immediately without a handshake.
    """
    (client_read, client_write), (server_read, server_write) = memory_channel_pair
    
    # Initialize with stateless=True
    session = ServerSession(
        read_stream=server_read,
        write_stream=server_write,
        init_options=InitializationOptions(
            server_name="stateless", server_version="1", capabilities=ServerCapabilities()
        ),
        stateless=True
    )
    
    async with session, client_write:
        assert session._initialization_state == InitializationState.Initialized

        # Send a request immediately (no initialize sent)
        req = JSONRPCRequest(
            jsonrpc="2.0", id=1, method="workflows/run", 
            params=RunWorkflowRequestParams(name="foo").model_dump(mode="json")
        )
        await client_write.send(SessionMessage(JSONRPCMessage(req)))
        
        # We verify it didn't raise "Received request before initialization"
        # by checking if the session yields the request for processing.
        async for responder in session.incoming_messages:
            if isinstance(responder, RequestResponder):
                assert responder.request.root.method == "workflows/run"
                break