"""Основной модуль для запуска приложения KAZrag."""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

# Устанавливаем переменные окружения для кэша моделей
models_dir = Path("./models")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(models_dir / "huggingface_cache", exist_ok=True)
os.makedirs(models_dir / "easyocr", exist_ok=True)
os.makedirs(models_dir / "fastembed", exist_ok=True)


# Устанавливаем переменные окружения для кэша моделей
# Path автоматически использует правильный разделитель пути для текущей ОС
os.environ["HF_HOME"] = str(models_dir / "huggingface_cache")
os.environ["FASTEMBED_CACHE_DIR"] = str(models_dir / "fastembed")
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Стабильность

# Импорт модулей для настройки приложения
from app.startup import startup_event_handler
from config.logging_config import setup_logging, get_logger

# Настройка логирования через централизованный модуль
setup_logging()
logger = get_logger(__name__)

# Загружаем переменные окружения из .env файла
load_dotenv()

# Устанавливаем токен HuggingFace для transformers, если он задан
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    logger.info("Приложение запускается")
    startup_event_handler()
    yield
    logger.info("Приложение останавливается")


def create_app_with_lifespan() -> FastAPI:
    """Create and configure the FastAPI application with lifespan."""
    from app.app_factory import create_app
    app = create_app()
    app.router.lifespan_context = lifespan
    return app


# Создание основного приложения FastAPI через фабрику
app = create_app_with_lifespan()


if __name__ == "__main__":
    logger.info("Начало запуска приложения")
    # Загрузка конфигурации при запуске с проверкой Qdrant будет выполнена в startup_event_handler
    logger.info("Обработчики событий зарегистрированы через lifespan")
    # Запуск сервера
    try:
        # Получаем IP адрес для доступа из локальной сети
        host = os.getenv("KAZRAG_HOST", "0.0.0.0")  # 0.0.0.0 - доступ из локальной сети
        port = int(os.getenv("KAZRAG_PORT", "8000"))

        if host == "0.0.0.0":
            logger.info(f"Запуск сервера на всех интерфейсах (доступ из локальной сети) на порту {port}")
            logger.info("Локальный доступ: http://127.0.0.1:8000")
            logger.info("Доступ из сети: http://[ваш-IP-адрес]:8000")
        else:
            logger.info(f"Запуск сервера на http://{host}:{port}")

        # Exclude common log file patterns and logs directory from reload watcher
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=True,
            reload_excludes=["*.log", "logs/*", "logs/*", "logs/*", "*.log.*", "*.log~", "file_tracking.json"],
            reload_dirs=["."]
        )
        logger.info("Сервер запущен")
    except Exception as e:
        logger.exception(f"Ошибка при запуске сервера: {e}")
        sys.exit(1)