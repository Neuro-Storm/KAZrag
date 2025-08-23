"""Centralized Qdrant client factory (sync + async helpers).

Provides a cached sync client and an async helper that creates the client in a thread
to avoid blocking the event loop.
"""
import asyncio
from functools import lru_cache
from typing import Optional

from qdrant_client import QdrantClient

from config.settings import load_config, Config


@lru_cache()
def _create_client(url: str) -> QdrantClient:
    return QdrantClient(url=url)


def get_qdrant_client(config: Optional[Config] = None) -> QdrantClient:
    """Return a cached synchronous QdrantClient for the provided config or loaded config."""
    if config is None:
        config = load_config()
    return _create_client(config.qdrant_url)


async def aget_qdrant_client(config: Optional[Config] = None) -> QdrantClient:
    """Async helper returning QdrantClient created in a thread to avoid blocking.

    Use when creating the client from async code that shouldn't block the event loop.
    """
    if config is None:
        config = load_config()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _create_client, config.qdrant_url)
