"""FastAPI приложение для поиска по документам."""

import logging

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from config.settings import Config
from core.qdrant.qdrant_collections import get_cached_collections

# Qdrant client is provided via dependency injection (get_client)
from core.search.searcher import search_in_collection
from core.utils.dependencies import get_client, get_config
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
        from pathlib import Path
        # Используем абсолютный путь от корня проекта
        templates_dir = Path(__file__).parent / "templates"
        _templates = Jinja2Templates(directory=str(templates_dir))
    return _templates


@app.get("/", response_class=HTMLResponse)
async def get_search_page(request: Request, config: Config = Depends(get_config), client = Depends(get_client)):
    """Отображает главную страницу поиска."""
    request_id = get_request_id(request)
    try:
        collections = get_cached_collections(client=client)
        logger.info(f"[{request_id}] Search page loaded successfully")
        return get_templates().TemplateResponse("index.html", {
            "request": request, 
            "results": [], 
            "collections": collections, 
            "selected_collection": config.search_default_collection,  # Добавляем глобальную настройку
            "k": config.search_default_k,  # Добавляем глобальную настройку
            "selected_search_device": config.search_default_device,  # Добавляем глобальную настройку
            "selected_search_type": config.search_default_type,  # Добавляем глобальную настройку
            "use_reranker": config.search_default_use_reranker,  # Добавляем глобальную настройку
            "config": config
        })
    except Exception as e:
        logger.error(f"[{request_id}] Error loading search page: {str(e)}")
        # Return page with error message
        collections = []
        return get_templates().TemplateResponse("index.html", {
            "request": request, 
            "results": [], 
            "collections": collections, 
            "selected_collection": config.search_default_collection,
            "k": config.search_default_k,
            "selected_search_device": config.search_default_device,
            "selected_search_type": config.search_default_type,
            "use_reranker": config.search_default_use_reranker,
            "config": config,
            "error": "Ошибка загрузки страницы поиска"
        })


@app.post("/", response_class=HTMLResponse)
async def handle_search(
    request: Request,
    query: str = Form(...),
    # collection: str = Form(...),  # Теперь используем глобальную настройку
    # search_device: str = Form("cpu"),  # Теперь используем глобальную настройку
    # k: int = Form(None),  # Теперь используем глобальную настройку
    # search_type: str = Form("dense"),  # Теперь используем глобальную настройку
    # use_reranker: bool = Form(True),  # Теперь используем глобальную настройку
    filter_author: str = Form(None),  # Фильтр по автору
    filter_source: str = Form(None),  # Фильтр по источнику
    filter_file_extension: str = Form(None),  # Фильтр по типу файла
    metadata_filter: str = Form(None),  # Произвольный фильтр по метаданным
    config: Config = Depends(get_config),
    client = Depends(get_client)
):
    """Обрабатывает поисковый запрос."""
    request_id = get_request_id(request)
    
    # Используем глобальные настройки из конфигурации
    collection = config.search_default_collection
    search_device = config.search_default_device
    k = config.search_default_k
    search_type = config.search_default_type
    use_reranker = config.search_default_use_reranker  # Используем глобальную настройку
    
    logger.info(f"[{request_id}] Search request received: query='{query}', collection='{collection}'")
    
    try:
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
        
        # Сохраняем оригинальную настройку reranker
        original_reranker_enabled = config.reranker_enabled
        
        # Если глобальная настройка reranker отключена, временно отключаем его
        if not use_reranker:
            config.reranker_enabled = False
        
        # Определяем режим поиска - если это sparse или hybrid, используем override
        search_mode_override = None
        if search_type in ["dense", "sparse", "hybrid"]:
            search_mode_override = search_type

        results, error = await search_in_collection(query, collection, search_device, k, 
                                                  hybrid=hybrid, 
                                                  search_mode_override=search_mode_override,
                                                  metadata_filter=metadata_filter_dict, 
                                                  client=client)
        
        # Восстанавливаем оригинальную настройку reranker
        config.reranker_enabled = original_reranker_enabled
        
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
            "use_reranker": use_reranker,
            "filter_author": filter_author,
            "filter_source": filter_source,
            "filter_file_extension": filter_file_extension,
            "metadata_filter": metadata_filter,
            "config": config
        }
        
        # Если есть ошибка, добавляем её в контекст шаблона
        if error:
            logger.warning(f"[{request_id}] Search returned error: {error}")
            template_context["error"] = error
        else:
            logger.info(f"[{request_id}] Search completed successfully with {len(results)} results")
            
            # Генерация RAG-ответа, если включено
            rag_response = None
            logger.info(f"[{request_id}] RAG enabled: {config.rag_enabled}, number of results: {len(results) if results else 0}")
            if config.rag_enabled and results:
                from core.search.searcher import generate_rag_response
                rag_response = generate_rag_response(query, results, config)
                logger.info(f"[{request_id}] RAG response generated: {rag_response is not None}, length: {len(rag_response) if rag_response else 0}")
                template_context["rag_response"] = rag_response
            else:
                logger.info(f"[{request_id}] RAG skipped - enabled: {config.rag_enabled}, has results: {bool(results)}")

        # В любом случае добавляем конфиг в контекст (если его еще нет)
        if "config" not in template_context:
            template_context["config"] = config
        
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
            "error": "Произошла ошибка при выполнении поиска",
            "config": config
        }
        return get_templates().TemplateResponse("index.html", template_context)