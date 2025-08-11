"""FastAPI приложение для поиска по документам."""

import logging
from fastapi import FastAPI, Form, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from qdrant_client import QdrantClient

from core.searcher import search_in_collection, get_qdrant_client
from core.qdrant_collections import get_cached_collections
from config.settings import load_config, Config
from core.dependencies import get_config

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация FastAPI приложения и шаблонов
app = FastAPI()
templates = Jinja2Templates(directory="web/templates")


@app.get("/", response_class=HTMLResponse)
async def get_search_page(request: Request, config: Config = Depends(get_config)):
    """Отображает главную страницу поиска."""
    collections = get_cached_collections()
    return templates.TemplateResponse("index.html", {
        "request": request, "results": [], "collections": collections
    })


@app.post("/", response_class=HTMLResponse)
async def handle_search(
    request: Request,
    query: str = Form(...),
    collection: str = Form(...),
    search_device: str = Form("cpu"),
    k: int = Form(5),
    config: Config = Depends(get_config)
):
    """Обрабатывает поисковый запрос."""
    results, error = search_in_collection(query, collection, search_device, k)
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    
    template_context = {
        "request": request,
        "results": results,
        "collections": collections,
        "selected_collection": collection,
        "query": query,
        "k": k,
        "selected_search_device": search_device
    }
    
    # Если есть ошибка, добавляем её в контекст шаблона
    if error:
        template_context["error"] = error
    
    return templates.TemplateResponse("index.html", template_context)