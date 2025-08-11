"""Модуль для зависимостей FastAPI."""

from config.settings import load_config, Config


def get_config() -> Config:
    """Зависимость для получения конфигурации."""
    return load_config()