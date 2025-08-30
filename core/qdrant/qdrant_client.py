"""Centralized Qdrant client factory (sync + async helpers).

Provides a cached sync client and an async helper that creates the client in a thread
to avoid blocking the event loop.
"""
import asyncio
from functools import lru_cache
from typing import Optional

from qdrant_client import QdrantClient

from config.config_manager import ConfigManager
from config.settings import Config

# Get singleton instance of ConfigManager
config_manager = ConfigManager.get_instance()


@lru_cache
def _create_client(
    url: str, 
    timeout: Optional[int] = None,
    prefer_grpc: bool = False
) -> QdrantClient:
    """Create a QdrantClient with specified parameters.
    
    Args:
        url: Qdrant server URL
        timeout: Connection timeout in seconds
        prefer_grpc: Whether to prefer gRPC over REST API
        
    Returns:
        QdrantClient: Configured Qdrant client
    """
    # Get default config for fallback values
    config_manager.get()
    
    # Use provided timeout or default from config
    client_timeout = timeout if timeout is not None else 10
    
    return QdrantClient(
        url=url,
        timeout=client_timeout,
        prefer_grpc=prefer_grpc
    )


def get_qdrant_client(
    config: Optional[Config] = None,
    timeout: Optional[int] = None,
    prefer_grpc: bool = False
) -> QdrantClient:
    """Return a cached synchronous QdrantClient for the provided config or loaded config.
    
    Args:
        config: Configuration object (optional)
        timeout: Connection timeout in seconds (optional)
        prefer_grpc: Whether to prefer gRPC over REST API (optional)
        
    Returns:
        QdrantClient: Configured Qdrant client
    """
    if config is None:
        config = config_manager.get()
    
    # Use provided timeout or get from config
    client_timeout = timeout if timeout is not None else config.qdrant_retry_wait_time
    
    return _create_client(
        url=config.qdrant_url,
        timeout=client_timeout,
        prefer_grpc=prefer_grpc
    )


async def aget_qdrant_client(
    config: Optional[Config] = None,
    timeout: Optional[int] = None,
    prefer_grpc: bool = False
) -> QdrantClient:
    """Async helper returning QdrantClient created in a thread to avoid blocking.

    Use when creating the client from async code that shouldn't block the event loop.
    
    Args:
        config: Configuration object (optional)
        timeout: Connection timeout in seconds (optional)
        prefer_grpc: Whether to prefer gRPC over REST API (optional)
        
    Returns:
        QdrantClient: Configured Qdrant client
    """
    if config is None:
        config = config_manager.get()
    
    # Use provided timeout or get from config
    client_timeout = timeout if timeout is not None else config.qdrant_retry_wait_time
    
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, 
        lambda: _create_client(
            url=config.qdrant_url,
            timeout=client_timeout,
            prefer_grpc=prefer_grpc
        )
    )