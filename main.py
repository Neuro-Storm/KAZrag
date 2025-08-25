"""Основной модуль для запуска приложения KAZrag."""

import uvicorn
import logging
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Импорт модулей для настройки приложения
from app_factory import create_app
from logging_config import setup_logging
from startup import startup_event_handler

# Настройка логирования через централизованный модуль
setup_logging()
logger = logging.getLogger(__name__)

# Загружаем переменные окружения из .env файла
load_dotenv()

# Проверяем и создаем директорию для кэша моделей fastembed, если она задана
fastembed_cache_dir = os.environ.get('FASTEMBED_CACHE_DIR')
if fastembed_cache_dir:
    fastembed_cache_path = Path(fastembed_cache_dir)
    fastembed_cache_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"FastEmbed cache directory: {fastembed_cache_path.absolute()}")

# Создание основного приложения FastAPI через фабрику
app = create_app()

# Register a middleware to generate request IDs
@app.middleware("http")
async def add_request_id(request, call_next):
    import uuid
    from fastapi import Request
    
    # Generate a unique request ID and attach it to the request state
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Process the request
    response = await call_next(request)
    
    return response

# Add favicon.ico route to prevent 404 errors
@app.get("/favicon.ico")
async def favicon():
    """Favicon route - returns 204 No Content to prevent 404 errors"""
    from fastapi.responses import Response
    return Response(status_code=204)

# Регистрация обработчиков событий запуска и остановки
@app.on_event("startup")
async def startup_event():
    """Handle application startup event."""
    startup_event_handler()

if __name__ == "__main__":
    logger.info("Начало запуска приложения")
    # Загрузка конфигурации при запуске с проверкой Qdrant
    try:
        from config.settings import load_config
        logger.info("Загрузка конфигурации")
        load_config()
        logger.info("Конфигурация успешно загружена")
    except RuntimeError as e:
        logger.error(f"Ошибка при загрузке конфигурации: {e}")
        sys.exit(1)
    # Запуск сервера
    try:
        logger.info("Запуск сервера на http://127.0.0.1:8000")
        # Exclude common log file patterns and logs directory from reload watcher
        uvicorn.run(
            "main:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            reload_excludes=["*.log", "logs/*", "*.log.*", "*.log~"]
        )
        logger.info("Сервер запущен")
    except Exception as e:
        logger.exception(f"Ошибка при запуске сервера: {e}")
        sys.exit(1)