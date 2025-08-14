"""Основной модуль для запуска приложения KAZrag."""

import uvicorn
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
import os
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загружаем переменные окружения из .env файла
load_dotenv()

# Импорт приложений
from web.search_app import app as search_app
from web.admin_app import app as admin_app

# Создание основного приложения FastAPI
app = FastAPI(title="KAZrag", description="Поисковая система на основе векторного поиска")

# Добавление CORS middleware
# Получаем разрешенные источники из переменной окружения или используем "*"
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins != "*":
    allowed_origins = allowed_origins.split(",")

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
    return RedirectResponse(url="/api/admin/settings")

if __name__ == "__main__":
    # Загрузка конфигурации при запуске с проверкой Qdrant
    try:
        from config.settings import load_config
        load_config()
    except RuntimeError as e:
        logger.error(f"Ошибка при загрузке конфигурации: {e}")
        exit(1)
    # Запуск сервера
    try:
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    except Exception as e:
        logger.error(f"Ошибка при запуске сервера: {e}")