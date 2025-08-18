"""Основной модуль для запуска приложения KAZrag."""

import uvicorn
import logging
import sys
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
import os
from pathlib import Path
from dotenv import load_dotenv

# Централизованная настройка логирования
import os
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(LOG_DIR / 'app.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
# Reduce verbosity of the file watcher to avoid log-change loops
logging.getLogger('watchfiles').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Загружаем переменные окружения из .env файла
load_dotenv()

# Проверяем и создаем директорию для кэша моделей fastembed, если она задана
fastembed_cache_dir = os.environ.get('FASTEMBED_CACHE_DIR')
if fastembed_cache_dir:
    fastembed_cache_path = Path(fastembed_cache_dir)
    fastembed_cache_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"FastEmbed cache directory: {fastembed_cache_path.absolute()}")

# Импорт приложений
from web.search_app import app as search_app
from web.admin_app import app as admin_app

# Создание основного приложения FastAPI
app = FastAPI(title="KAZrag", description="Поисковая система на основе векторного поиска")

# Добавление CORS middleware
# Нормализуем ALLOWED_ORIGINS: '*' -> ['*'], иначе список origin'ов разделённых запятыми
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*") or "*"
if allowed_origins_env.strip() == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [o.strip() for o in allowed_origins_env.split(',') if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтирование приложений
app.mount("/api/search", search_app)
app.mount("/api/admin", admin_app)

# Добавляем корневой маршрут для перенаправления на страницу поиска
@app.get("/", response_class=RedirectResponse)
async def root():
    """Корневой маршрут - перенаправляет на страницу поиска"""
    return RedirectResponse(url="/api/search/")

# Добавляем маршрут для настроек, который перенаправляет на админку
@app.get("/settings", response_class=RedirectResponse)
async def settings_redirect():
    """Маршрут для настроек - перенаправляет на админку"""
    return RedirectResponse(url="/api/admin/admin/settings")

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
            reload=False
        )
        logger.info("Сервер запущен")
    except Exception as e:
        logger.exception(f"Ошибка при запуске сервера: {e}")
        sys.exit(1)