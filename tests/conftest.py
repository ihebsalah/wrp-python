# tests/conftest.py
import logging
import pytest
import anyio

# ------------------------------------------------------------------------------
# 1. Global Configuration
# ------------------------------------------------------------------------------

@pytest.fixture(scope="session")
def anyio_backend():
    """
    Tells pytest to use 'asyncio' as the backend for anyio tests.
    WRP relies heavily on anyio, and this ensures compatibility with libs like httpx.
    """
    return "asyncio"


@pytest.fixture(autouse=True)
def setup_test_logging(caplog):
    """
    Automatically captures logging at DEBUG level for every test.
    If a test fails, pytest will show the logs (WRP logs a lot of useful debug info).
    """
    caplog.set_level(logging.DEBUG)


# ------------------------------------------------------------------------------
# 2. Shared Transport Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
async def memory_channel_pair():
    """
    Creates a bidirectional memory stream pair for testing transport layers.
    
    This fixture creates two pairs of streams to simulate a full-duplex
    connection. It uses a generator with a teardown phase to ensure that all
    streams are closed properly after a test, preventing hangs.

    Yields:
        tuple: A pair of tuples for client and server sides:
               `((client_read, client_write), (server_read, server_write))`
    """
    # Stream A: Traffic going TO the Server (Client writes, Server reads)
    client_write, server_read = anyio.create_memory_object_stream(100)
    
    # Stream B: Traffic going TO the Client (Server writes, Client reads)
    server_write, client_read = anyio.create_memory_object_stream(100)

    client_side = (client_read, client_write)
    server_side = (server_read, server_write)
    
    yield client_side, server_side

    # Teardown: Close all streams to ensure any background tasks listening on them
    # can exit gracefully. The timeout prevents tests from blocking indefinitely.
    with anyio.move_on_after(1, shield=True):
        await client_write.aclose()
        await server_write.aclose()
        await client_read.aclose()
        await server_read.aclose()