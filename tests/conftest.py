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
def memory_channel_pair():
    """
    Creates a bidirectional memory stream pair to simulate a transport layer 
    without using HTTP or Stdio.
    
    This is essential for testing 'BaseSession', 'ClientSession', and 'ServerSession' 
    in isolation (Unit tests).

    Visual flow:
      [Client Side] --(client_write)--> Stream A --(server_read)--> [Server Side]
      [Client Side] <--(client_read)-- Stream B <--(server_write)-- [Server Side]

    Returns:
        tuple: (
            (client_read_stream, client_write_stream), 
            (server_read_stream, server_write_stream)
        )
    """
    # Stream A: Traffic going TO the Server
    server_read_stream, client_write_stream = anyio.create_memory_object_stream(100)
    
    # Stream B: Traffic going TO the Client
    client_read_stream, server_write_stream = anyio.create_memory_object_stream(100)

    client_side = (client_read_stream, client_write_stream)
    server_side = (server_read_stream, server_write_stream)

    return client_side, server_side