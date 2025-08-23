"""FastAPI приложение для поиска по документам."""

import logging
from fastapi import FastAPI, Form, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
# Qdrant client is provided via dependency injection (get_client)

from core.searcher import search_in_collection
from core.qdrant_collections import get_cached_collections
from config.settings import Config
from core.dependencies import get_config, get_client

logger = logging.getLogger(__name__)

# Инициализация FastAPI приложения и шаблонов
app = FastAPI()
templates = Jinja2Templates(directory="web/templates")


@app.get("/", response_class=HTMLResponse)
async def get_search_page(request: Request, config: Config = Depends(get_config), client = Depends(get_client)):
    """Отображает главную страницу поиска."""
    collections = get_cached_collections(client=client)
    return templates.TemplateResponse("index.html", {
        "request": request, "results": [], "collections": collections
    })


@app.post("/", response_class=HTMLResponse)
async def handle_search(
    request: Request,
    query: str = Form(...),
    collection: str = Form(...),
    search_device: str = Form("cpu"),
    k: int = Form(None),  # Используем None как значение по умолчанию
    search_type: str = Form("dense"),  # Новый параметр для типа поиска
    hybrid: bool = Form(False),  # Для обратной совместимости
    config: Config = Depends(get_config),
    client = Depends(get_client)
):
    """Обрабатывает поисковый запрос."""
    # Если k не указано, используем значение из конфигурации
    if k is None:
        k = config.search_default_k
    
    # Определяем тип поиска
    if search_type == "hybrid":
        hybrid = True
    elif search_type == "sparse":
        # Для sparse поиска hybrid должен быть False
        hybrid = False
    else:  # dense
        hybrid = False
    
    results, error = await search_in_collection(query, collection, search_device, k, hybrid=hybrid, client=client)
    collections = get_cached_collections(client=client)
    
    template_context = {
        "request": request,
        "results": results,
        "collections": collections,
        "selected_collection": collection,
        "query": query,
        "k": k,
        "selected_search_device": search_device,
        "hybrid": hybrid,
        "selected_search_type": search_type
    }
    
    # Если есть ошибка, добавляем её в контекст шаблона
    if error:
        template_context["error"] = error
    
    return templates.TemplateResponse("index.html", template_context)