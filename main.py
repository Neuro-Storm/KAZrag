"""Основной модуль для запуска приложения KAZrag."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Импорт приложений
from web.search_app import app as search_app
from web.admin_app import app as admin_app

# Создание основного приложения FastAPI
app = FastAPI(title="KAZrag", description="Поисковая система на основе векторного поиска")

# Добавление CORS middleware (опционально, если потребуется)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В production лучше указать конкретные origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтирование приложений
app.mount("/api/search", search_app)
app.mount("/api/admin", admin_app)

# Объединение маршрутов для удобства доступа через корневой путь
# Это позволяет использовать пути как раньше, без префиксов
app.routes.extend(search_app.routes)
app.routes.extend(admin_app.routes)

if __name__ == "__main__":
    # Загрузка конфигурации при запуске
    from config.settings import load_config
    load_config()
    # Запуск сервера
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)