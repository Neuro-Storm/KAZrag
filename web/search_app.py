"""FastAPI приложение для поиска по документам."""

import logging
import uuid
from fastapi import APIRouter, Form, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
# Qdrant client is provided via dependency injection (get_client)

from core.search.searcher import search_in_collection
from core.qdrant.qdrant_collections import get_cached_collections
from config.settings import Config
from core.utils.dependencies import get_config, get_client
from core.utils.exception_handlers import get_request_id

logger = logging.getLogger(__name__)

# Инициализация APIRouter
app = APIRouter()

# Lazy initialization of templates
_templates = None


def get_templates():
    """Get or create Jinja2Templates instance."""
    global _templates
    if _templates is None:
        _templates = Jinja2Templates(directory="web/templates")
    return _templates


@app.get("/", response_class=HTMLResponse)
async def get_search_page(request: Request, config: Config = Depends(get_config), client = Depends(get_client)):
    """Отображает главную страницу поиска."""
    request_id = get_request_id(request)
    try:
        collections = get_cached_collections(client=client)
        logger.info(f"[{request_id}] Search page loaded successfully")
        return get_templates().TemplateResponse("index.html", {
            "request": request, "results": [], "collections": collections, "config": config
        })
    except Exception as e:
        logger.error(f"[{request_id}] Error loading search page: {str(e)}")
        # Return page with error message
        collections = []
        return get_templates().TemplateResponse("index.html", {
            "request": request, 
            "results": [], 
            "collections": collections, 
            "config": config,
            "error": "Ошибка загрузки страницы поиска"
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
    filter_author: str = Form(None),  # Фильтр по автору
    filter_source: str = Form(None),  # Фильтр по источнику
    filter_file_extension: str = Form(None),  # Фильтр по типу файла
    metadata_filter: str = Form(None),  # Произвольный фильтр по метаданным
    config: Config = Depends(get_config),
    client = Depends(get_client)
):
    """Обрабатывает поисковый запрос."""
    request_id = get_request_id(request)
    logger.info(f"[{request_id}] Search request received: query='{query}', collection='{collection}'")
    
    try:
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
        
        # Собираем фильтр по метаданным из отдельных полей
        metadata_filter_dict = {}
        
        # Добавляем фильтры из отдельных полей
        if filter_author:
            metadata_filter_dict["author"] = filter_author
        if filter_source:
            metadata_filter_dict["source"] = filter_source
        if filter_file_extension:
            metadata_filter_dict["file_extension"] = filter_file_extension
        
        # Если есть произвольный фильтр, парсим его и объединяем с остальными
        if metadata_filter:
            try:
                # Парсим строку фильтра в словарь
                import json
                custom_filter = json.loads(metadata_filter)
                # Объединяем с существующими фильтрами
                metadata_filter_dict.update(custom_filter)
            except Exception as e:
                logger.warning(f"[{request_id}] Ошибка при парсинге произвольного фильтра по метаданным: {e}")
        
        # Если нет фильтров, устанавливаем None
        if not metadata_filter_dict:
            metadata_filter_dict = None
        
        results, error = await search_in_collection(query, collection, search_device, k, hybrid=hybrid, 
                                                  metadata_filter=metadata_filter_dict, client=client)
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
            "selected_search_type": search_type,
            "filter_author": filter_author,
            "filter_source": filter_source,
            "filter_file_extension": filter_file_extension,
            "metadata_filter": metadata_filter
        }
        
        # Если есть ошибка, добавляем её в контекст шаблона
        if error:
            logger.warning(f"[{request_id}] Search returned error: {error}")
            template_context["error"] = error
        else:
            logger.info(f"[{request_id}] Search completed successfully with {len(results)} results")
        
        # Проверяем, является ли запрос AJAX запросом
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
        
        if is_ajax:
            # Для AJAX запросов возвращаем только результаты поиска
            template_response = get_templates().TemplateResponse("search_results.html", template_context)
            # Убедимся, что у нас правильный content-type для HTML
            template_response.headers["Content-Type"] = "text/html; charset=utf-8"
            return template_response
        else:
            # Для обычных запросов возвращаем полную страницу
            return get_templates().TemplateResponse("index.html", template_context)
            
    except Exception as e:
        logger.exception(f"[{request_id}] Unhandled error during search: {str(e)}")
        # Return error to user
        collections = get_cached_collections(client=client)
        template_context = {
            "request": request,
            "results": [],
            "collections": collections,
            "selected_collection": collection,
            "query": query,
            "error": "Произошла ошибка при выполнении поиска"
        }
        return get_templates().TemplateResponse("index.html", template_context)