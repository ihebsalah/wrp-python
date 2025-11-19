# tests/unit/shared/test_base_session.py
import pytest
import anyio
from datetime import timedelta
from wrp.client.session import ClientSession
from wrp.server.session import ServerSession
from wrp.server.models import InitializationOptions
from wrp.shared.exceptions import WrpError
from wrp.types import (
    ServerCapabilities,
    Implementation,
    ClientNotification,
    InitializeRequest,
    PingRequest,
    ServerResult,
    InitializeResult,
    EmptyResult,
    ErrorData,
    INTERNAL_ERROR,
    LoggingMessageNotificationParams,
    ClientRequest,
    CancelledNotification,
)

@pytest.mark.anyio
async def test_session_initialization_and_handshake(memory_channel_pair):
    """
    Verifies that a ClientSession can successfully initialize a connection
    with a ServerSession over memory streams.
    """
    # 1. Setup Fixtures (Client <-> Server connected via memory streams)
    (client_read, client_write), (server_read, server_write) = memory_channel_pair

    # 2. Configure Server
    init_options = InitializationOptions(
        server_name="test-server",
        server_version="1.0.0",
        capabilities=ServerCapabilities(
            resources=None,
            workflows=None,
        )
    )

    server = ServerSession(
        read_stream=server_read,
        write_stream=server_write,
        init_options=init_options,
    )

    # 3. Configure Client
    client = ClientSession(
        read_stream=client_read,
        write_stream=client_write,
        client_info=Implementation(name="test-client", version="1.0.0")
    )

    # 4. Run interactions within context managers (starts background receive loops)
    async with client, server:
        # Create a "Brain" for the server to process incoming requests.
        # This loop runs in the background and responds to client messages.
        async def mock_server_loop():
           async for responder in server.incoming_messages:
               if isinstance(responder, Exception):
                   continue

               # Notifications do not require a response
               if isinstance(responder, ClientNotification):
                   continue

               # The responder MUST be used as a context manager to handle lifecycle/cancellation
               with responder:
                   req = responder.request.root
                   if isinstance(req, InitializeRequest):
                       # Respond to the "initialize" request with server details
                       await responder.respond(ServerResult(
                           InitializeResult(
                               protocolVersion="2025-11-01",
                               capabilities=init_options.capabilities,
                               serverInfo=Implementation(
                                   name=init_options.server_name,
                                   version=init_options.server_version
                               )
                           )
                       ))
                   elif isinstance(req, PingRequest):
                       # Respond to "ping" requests with an empty result
                       await responder.respond(ServerResult(EmptyResult()))

        # Run the mock server loop in the background while the client makes requests
        async with anyio.create_task_group() as tg:
            tg.start_soon(mock_server_loop)

            # --- Phase A: Initialization ---
            # Client sends "initialize" -> Server responds with capabilities
            init_result = await client.initialize()

            assert init_result.protocolVersion == "2025-11-01"
            assert init_result.serverInfo.name == "test-server"

            # Verify server state moved to Initialized
            # (The client sends 'notifications/initialized' automatically inside .initialize())
            # We yield briefly to let the server process the notification.
            await anyio.sleep(0.01)

            # --- Phase B: Ping (Round trip) ---
            # Client sends "ping" -> Server responds "EmptyResult"
            await client.send_ping()

            # Server sends "ping" -> Client responds "EmptyResult"
            await server.send_ping()

            # Stop the background server loop cleanly
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_session_error_handling(memory_channel_pair):
    """
    Verifies that if the Server sends a JSON-RPC Error, the Client raises a WrpError.
    """
    (client_read, client_write), (server_read, server_write) = memory_channel_pair

    # Setup (Simplified for error test)
    client = ClientSession(client_read, client_write)
    server = ServerSession(server_read, server_write, InitializationOptions(
        server_name="err-server", server_version="1.0", capabilities=ServerCapabilities()
    ))

    async with client, server:
        async def error_server_loop():
            async for responder in server.incoming_messages:
                if isinstance(responder, Exception): continue

                # Ignore notifications so the type checker knows responder is a RequestResponder
                if isinstance(responder, ClientNotification): continue

                with responder:
                    # Simulate a server-side crash or logic error
                    await responder.respond(ErrorData(
                        code=INTERNAL_ERROR,
                        message="Something went wrong server-side",
                        data={"detail": "stack trace here"}
                    ))

        async with anyio.create_task_group() as tg:
            tg.start_soon(error_server_loop)

            # Client expects a result, but Server sends Error
            # We use send_ping as a generic request carrier here
            with pytest.raises(WrpError) as exc_info:
                await client.send_ping()

            assert exc_info.value.error.code == INTERNAL_ERROR
            assert exc_info.value.error.message == "Something went wrong server-side"
            assert exc_info.value.error.data == {"detail": "stack trace here"}

            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_session_notifications(memory_channel_pair):
    """
    Verifies one-way notifications (Fire and Forget).
    """
    (client_read, client_write), (server_read, server_write) = memory_channel_pair

    # We need a way to check if the callback was triggered
    captured_logs = []

    async def log_callback(params: LoggingMessageNotificationParams):
        captured_logs.append(params)

    # Initialize client with a specific notification handler
    client = ClientSession(
        client_read, client_write,
        logging_callback=log_callback
    )
    server = ServerSession(server_read, server_write, InitializationOptions(
        server_name="notif-server", server_version="1.0", capabilities=ServerCapabilities()
    ))

    async with client, server:
        # Server sends a log message
        await server.send_log_message(
            level="info",
            data="System is booting",
            logger="system.boot"
        )

        # Give the event loop a moment to process the message
        # (Notifications are async and don't block send_log_message)
        with anyio.move_on_after(1):
            while len(captured_logs) == 0:
                await anyio.sleep(0.01)

        assert len(captured_logs) == 1
        assert captured_logs[0].level == "info"
        assert captured_logs[0].data == "System is booting"
        assert captured_logs[0].logger == "system.boot"


@pytest.mark.anyio
async def test_session_timeout(memory_channel_pair):
    """
    Verifies that the client raises a timeout error if the server takes too long.
    """
    (client_read, client_write), (server_read, server_write) = memory_channel_pair

    client = ClientSession(client_read, client_write)
    server = ServerSession(server_read, server_write, InitializationOptions(
        server_name="slow-server", server_version="1.0", capabilities=ServerCapabilities()
    ))

    async with client, server:
        async def slow_server_loop():
            async for responder in server.incoming_messages:
                if isinstance(responder, (Exception, ClientNotification)): continue
                with responder:
                    # Sleep longer than the client timeout
                    await anyio.sleep(0.2)
                    await responder.respond(ServerResult(EmptyResult()))

        async with anyio.create_task_group() as tg:
            tg.start_soon(slow_server_loop)

            # Client expects response in 0.1s, Server takes 0.2s
            with pytest.raises(WrpError) as exc_info:
                # We pass a specific timeout to the underlying send_request method
                await client.send_request(
                    ClientRequest(root=PingRequest()),
                    EmptyResult,
                    request_read_timeout_seconds=timedelta(seconds=0.1)
                )

            # Check that the error message indicates a timeout occurred
            assert "Timed out" in exc_info.value.error.message
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_session_progress(memory_channel_pair):
    """
    Verifies that progress notifications from Server are routed
    to the specific request's callback on the Client.
    """
    (client_read, client_write), (server_read, server_write) = memory_channel_pair

    client = ClientSession(client_read, client_write)
    server = ServerSession(server_read, server_write, InitializationOptions(
        server_name="prog-server", server_version="1.0", capabilities=ServerCapabilities()
    ))

    progress_updates = []
    async def on_progress(progress: float, total: float | None, message: str | None):
        progress_updates.append((progress, total))

    async with client, server:
        async def progress_server_loop():
            async for responder in server.incoming_messages:
                if isinstance(responder, (Exception, ClientNotification)): continue

                with responder:
                    # 1. Extract the progress token sent by client
                    req = responder.request.root
                    assert req.params is not None
                    assert req.params.meta is not None
                    token = req.params.meta.progressToken
                    assert token is not None

                    # 2. Send progress back to the client using the token
                    await server.send_progress_notification(token, 10, 100, "Starting")
                    await anyio.sleep(0.01)
                    await server.send_progress_notification(token, 50, 100, "Halfway")

                    # 3. Finish the request by sending the final response
                    await responder.respond(ServerResult(EmptyResult()))

        async with anyio.create_task_group() as tg:
            tg.start_soon(progress_server_loop)

            # Send request WITH progress callback to capture progress notifications
            await client.send_request(
                ClientRequest(root=PingRequest()),
                EmptyResult,
                progress_callback=on_progress
            )

            assert len(progress_updates) == 2
            assert progress_updates[0] == (10, 100)
            assert progress_updates[1] == (50, 100)

            tg.cancel_scope.cancel()