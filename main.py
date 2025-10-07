"""Основной модуль для запуска приложения KAZrag."""

import os
import sys
import unittest  # Явный импорт для PyInstaller
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

# Устанавливаем переменные окружения для кэша моделей
models_dir = Path("./models")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(models_dir / "huggingface_cache", exist_ok=True)
os.makedirs(models_dir / "easyocr", exist_ok=True)
os.makedirs(models_dir / "fastembed", exist_ok=True)

os.environ["TRANSFORMERS_CACHE"] = str(models_dir / "huggingface_cache")
os.environ["HF_HOME"] = str(models_dir / "huggingface_cache")
os.environ["FASTEMBED_CACHE_DIR"] = str(models_dir / "fastembed")
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Стабильность

# Импорт модулей для настройки приложения
from app.app_factory import create_app
from app.startup import startup_event_handler
from config.logging_config import setup_logging, get_logger, setup_intercept_handler

# Настройка логирования через централизованный модуль
setup_logging()
setup_intercept_handler()
logger = get_logger(__name__)

# Загружаем переменные окружения из .env файла
load_dotenv()

# Устанавливаем токен HuggingFace для transformers, если он задан
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# Создание основного приложения FastAPI через фабрику
app = create_app()





# Регистрация обработчиков событий запуска и остановки
@app.on_event("startup")
async def startup_event():
    """Handle application startup event."""
    startup_event_handler()

if __name__ == "__main__":
    logger.info("Начало запуска приложения")
    # Загрузка конфигурации при запуске с проверкой Qdrant
    try:
        logger.info("Попытка импорта ConfigManager")
        from config.config_manager import ConfigManager
        logger.info("ConfigManager успешно импортирован")
        logger.info("Загрузка конфигурации")
        config_manager = ConfigManager.get_instance()
        config_manager.get()
        logger.info("Конфигурация успешно загружена")
    except RuntimeError as e:
        logger.error(f"Ошибка при загрузке конфигурации: {e}")
        sys.exit(1)
    # Создание основного приложения FastAPI через фабрику
    logger.info("Создание приложения")
    app = create_app()
    logger.info("Приложение успешно создано")
    # Регистрация обработчиков событий запуска и остановки
    logger.info("Регистрация обработчиков событий")
    @app.on_event("startup")
    async def startup_event():
        """Handle application startup event."""
        startup_event_handler()
    logger.info("Обработчики событий зарегистрированы")
    # Запуск сервера
    try:
        logger.info("Запуск сервера на http://127.0.0.1:8000")
        # Exclude common log file patterns and logs directory from reload watcher
        uvicorn.run(
            "main:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            reload_excludes=["*.log", "logs/*", "logs/*", "logs/*", "*.log.*", "*.log~"],
            reload_dirs=["."]
        )
        logger.info("Сервер запущен")
    except Exception as e:
        logger.exception(f"Ошибка при запуске сервера: {e}")
        sys.exit(1)