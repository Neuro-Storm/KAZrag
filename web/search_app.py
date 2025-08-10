"""FastAPI приложение для поиска по документам."""

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from qdrant_client import QdrantClient

from core.searcher import search_in_collection, get_qdrant_client
from core.qdrant_collections import get_cached_collections
from config.settings import load_config

# Инициализация FastAPI приложения и шаблонов
app = FastAPI()
templates = Jinja2Templates(directory="web/templates")


@app.get("/", response_class=HTMLResponse)
async def get_search_page(request: Request):
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
    k: int = Form(5)
):
    """Обрабатывает поисковый запрос."""
    results = search_in_collection(query, collection, search_device, k)
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": results,
        "collections": collections,
        "selected_collection": collection,
        "query": query,
        "k": k,
        "selected_search_device": search_device
    })