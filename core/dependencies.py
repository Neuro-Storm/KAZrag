"""Модуль для зависимостей FastAPI."""

from fastapi import Depends
from qdrant_client import QdrantClient
from config.settings import load_config, Config


def get_config() -> Config:
    """Зависимость для получения конфигурации."""
    return load_config()


def get_client(config: Config = Depends(get_config)) -> QdrantClient:
    """Зависимость для получения клиента Qdrant."""
    return QdrantClient(url=config.qdrant_url)