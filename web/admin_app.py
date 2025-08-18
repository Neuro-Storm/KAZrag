"""FastAPI приложение для настроек и управления системой."""

import traceback
import os
import logging
from typing import Optional
from fastapi import FastAPI, Form, Request, Depends, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
# Qdrant client provided via dependency injection (core.dependencies.get_client)
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

import os
from pathlib import Path

# Проверяем и создаем директорию для кэша моделей fastembed, если она задана
fastembed_cache_dir = os.environ.get('FASTEMBED_CACHE_DIR')
if fastembed_cache_dir:
    fastembed_cache_path = Path(fastembed_cache_dir)
    fastembed_cache_path.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

from config.settings import load_config, save_config, Config
from core.indexer import run_indexing_logic
from core.file_converter import run_pdf_processing_from_config
from core.qdrant_collections import get_cached_collections, refresh_collections_cache
from core.dependencies import get_config, get_client

# Инициализация FastAPI приложения и шаблонов
app = FastAPI()
templates = Jinja2Templates(directory="web/templates")

# Настройка HTTP Basic Authentication
security = HTTPBasic()

# Получаем API ключ из переменной окружения
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")


def update_index_settings(form_data: dict, config: Config):
    """Обновляет настройки индексации."""
    model_changed = False
    device_changed = False
    
    # Валидация обязательных полей
    if form_data.get("folder_path") is None or not form_data["folder_path"].strip():
        raise HTTPException(400, detail="Путь к папке не может быть пустым")
    config.folder_path = form_data["folder_path"].strip()
    
    if form_data.get("collection_name") is None or not form_data["collection_name"].strip():
        raise HTTPException(400, detail="Имя коллекции не может быть пустым")
    config.collection_name = form_data["collection_name"].strip()
    
    if form_data.get("hf_model") is None or not form_data["hf_model"].strip():
        raise HTTPException(400, detail="Модель не может быть пустой")
    config.current_hf_model = form_data["hf_model"].strip()
    
    if form_data.get("chunk_size") is not None:
        try:
            config.chunk_size = int(form_data["chunk_size"])
        except ValueError:
            # Можно добавить flash сообщение об ошибке
            raise HTTPException(400, detail="Неверный размер чанка")
    if form_data.get("chunk_overlap") is not None:
        try:
            config.chunk_overlap = int(form_data["chunk_overlap"])
        except ValueError:
            # Можно добавить flash сообщение об ошибке
            raise HTTPException(400, detail="Неверное перекрытие чанков")
    if form_data.get("embedding_batch_size") is not None:
        try:
            config.embedding_batch_size = int(form_data["embedding_batch_size"])
        except ValueError:
            # Можно добавить flash сообщение об ошибке
            raise HTTPException(400, detail="Неверный размер батча для эмбеддинга")
    if form_data.get("indexing_batch_size") is not None:
        try:
            config.indexing_batch_size = int(form_data["indexing_batch_size"])
        except ValueError:
            # Можно добавить flash сообщение об ошибке
            raise HTTPException(400, detail="Неверный размер батча для индексации")
    
    # use_dense теперь bool
    # legacy fields and new explicit index flags
    config.use_dense_vectors = form_data.get("use_dense", False)
    # New explicit flags for index type
    config.index_dense = bool(form_data.get("index_dense", form_data.get("use_dense", False)))
    config.index_bm25 = bool(form_data.get("index_bm25", False))
    config.index_hybrid = bool(form_data.get("index_hybrid", form_data.get("use_hybrid", False)))
    # keep legacy flag for backward compatibility
    config.use_hybrid = form_data.get("use_hybrid", False)
    
    if form_data.get("device") is not None:
        config.device = form_data["device"]
        if form_data["device"] != config.device:
            device_changed = True
    
    config.is_indexed = False # Сбрасываем флаг индексации

    if form_data.get("hf_model") is not None and form_data["hf_model"] not in config.hf_model_history:
        config.hf_model_history.append(form_data["hf_model"])

    # Сброс кэшa embedder'а при смене модели или устройства
    if model_changed or device_changed:
        from core.embeddings import _dense_embedder_cache
        _dense_embedder_cache.clear()
        # Сброс кэша text_splitter при смене chunk_size или chunk_overlap
        from core.chunker import get_text_splitter
        get_text_splitter.cache_clear()
        # Принудительно очищаем GPU память, если используется CUDA
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def update_mineru_settings(form_data: dict, config: Config):
    """Обновляет настройки MinerU."""
    if form_data.get("mineru_input_pdf_dir") is not None: 
        config.mineru_input_pdf_dir = form_data["mineru_input_pdf_dir"]
    if form_data.get("mineru_output_md_dir") is not None: 
        config.mineru_output_md_dir = form_data["mineru_output_md_dir"]
    
    # Checkboxes теперь bool
    config.mineru_enable_formula_parsing = form_data.get("mineru_enable_formula_parsing", False)
    config.mineru_enable_table_parsing = form_data.get("mineru_enable_table_parsing", False)
    
    if form_data.get("mineru_model_source") is not None: 
        config.mineru_model_source = form_data["mineru_model_source"]
    
    # Optional поля
    if form_data.get("mineru_models_dir") is not None:
        mineru_models_dir = form_data["mineru_models_dir"].strip()
        if mineru_models_dir:
            config.mineru_models_dir = mineru_models_dir
    
    if form_data.get("mineru_backend") is not None: 
        config.mineru_backend = form_data["mineru_backend"]
    if form_data.get("mineru_method") is not None: 
        config.mineru_method = form_data["mineru_method"]
    if form_data.get("mineru_lang") is not None: 
        config.mineru_lang = form_data["mineru_lang"]
    
    # Optional поля
    if form_data.get("mineru_sglang_url") is not None:
        mineru_sglang_url = form_data["mineru_sglang_url"].strip()
        if mineru_sglang_url:
            config.mineru_sglang_url = mineru_sglang_url


def update_advanced_settings(form_data: dict, config: Config):
    """Обновляет дополнительные настройки."""
    # Настройки кэширования
    if form_data.get("config_cache_ttl") is not None:
        try:
            config.config_cache_ttl = int(form_data["config_cache_ttl"])
        except ValueError:
            raise HTTPException(400, detail="Неверное значение TTL кэша конфигурации")
    if form_data.get("qdrant_client_cache_ttl") is not None:
        try:
            config.qdrant_client_cache_ttl = int(form_data["qdrant_client_cache_ttl"])
        except ValueError:
            raise HTTPException(400, detail="Неверное значение TTL кэша клиента Qdrant")
    if form_data.get("collections_cache_ttl") is not None:
        try:
            config.collections_cache_ttl = int(form_data["collections_cache_ttl"])
        except ValueError:
            raise HTTPException(400, detail="Неверное значение TTL кэша коллекций")
    
    # Настройки GGUF моделей
    if form_data.get("gguf_model_n_ctx") is not None:
        try:
            config.gguf_model_n_ctx = int(form_data["gguf_model_n_ctx"])
        except ValueError:
            raise HTTPException(400, detail="Неверный размер контекста для GGUF моделей")
    
    # Настройки поиска
    if form_data.get("search_default_k") is not None:
        try:
            config.search_default_k = int(form_data["search_default_k"])
        except ValueError:
            raise HTTPException(400, detail="Неверное количество результатов поиска по умолчанию")
    
    # Настройки подключения к Qdrant
    if form_data.get("qdrant_url") is not None:
        config.qdrant_url = form_data["qdrant_url"].strip()
    if form_data.get("qdrant_retry_attempts") is not None:
        try:
            config.qdrant_retry_attempts = int(form_data["qdrant_retry_attempts"])
        except ValueError:
            raise HTTPException(400, detail="Неверное количество попыток повтора для подключения к Qdrant")
    if form_data.get("qdrant_retry_wait_time") is not None:
        try:
            config.qdrant_retry_wait_time = int(form_data["qdrant_retry_wait_time"])
        except ValueError:
            raise HTTPException(400, detail="Неверное время ожидания между попытками повтора")
    
    # Настройки индексации документов
    if form_data.get("memory_threshold") is not None:
        try:
            config.memory_threshold = int(form_data["memory_threshold"])
        except ValueError:
            raise HTTPException(400, detail="Неверный порог памяти")
    if form_data.get("indexing_default_batch_size") is not None:
        try:
            config.indexing_default_batch_size = int(form_data["indexing_default_batch_size"])
        except ValueError:
            raise HTTPException(400, detail="Неверный размер батча для индексации документов по умолчанию")
    if form_data.get("sparse_embedding") is not None:
        config.sparse_embedding = form_data["sparse_embedding"].strip()
    
    # Настройки MinerU (дополнительные)
    if form_data.get("mineru_subprocess_timeout") is not None:
        try:
            config.mineru_subprocess_timeout = int(form_data["mineru_subprocess_timeout"])
        except ValueError:
            raise HTTPException(400, detail="Неверный таймаут subprocess вызова mineru")


async def verify_admin_access(credentials: HTTPBasicCredentials = Depends(security)):
    """Проверяет учетные данные для доступа к админке."""
    # Если API ключ не установлен в .env, разрешаем доступ без аутентификации
    if not ADMIN_API_KEY:
        logger.warning("ADMIN_API_KEY not set, access denied")
        raise HTTPException(
            status_code=401,
            detail="API key not set",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    # Проверяем учетные данные
    # В данном случае используем API ключ как пароль, имя пользователя может быть любым
    if credentials.password == ADMIN_API_KEY:
        return credentials.username
    
    # В противном случае, возвращаем ошибку
    raise HTTPException(
        status_code=401,
        detail="Неверные учетные данные",
        headers={"WWW-Authenticate": "Basic"},
    )


@app.get("/settings", response_class=HTMLResponse)
async def get_settings_page(request: Request, status: str = None, username: str = Depends(verify_admin_access), config: Config = Depends(get_config), client = Depends(get_client)):
    """Отображает страницу настроек."""
    logger.debug(f"Загружена конфигурация: {config.dict()}")
    collections = get_cached_collections(client=client)
    # Получаем директорию кэша fastembed из переменных окружения
    fastembed_cache_dir = os.environ.get('FASTEMBED_CACHE_DIR')
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "config": config,
        "collections": collections,
        "delete_status": status,
        "fastembed_cache_dir": fastembed_cache_dir
    })


@app.post("/settings/delete-collection", response_class=RedirectResponse)
async def delete_collection(request: Request, collection_name: str = Form(...), username: str = Depends(verify_admin_access), config: Config = Depends(get_config), client = Depends(get_client)):
    """Удаляет указанную коллекцию из Qdrant."""
    status_msg = ""
    try:
        if not collection_name:
             status_msg = "error_no_collection_selected"
        else:
            client.delete_collection(collection_name)
            await refresh_collections_cache(client=client)  # Обновляем кэш после удаления
            status_msg = f"deleted_{collection_name}"
    except Exception as e:
        logger.exception(f"Ошибка при удалении коллекции '{collection_name}': {e}")
        status_msg = f"delete_error"
    return RedirectResponse(url=f"/settings?status={status_msg}", status_code=303)


# Обновление настроек приложения с разделением на типы (индексация/MinerU/Дополнительные)
@app.post("/update-settings", response_class=RedirectResponse)
async def update_settings(
    request: Request,
    # --- Поля индексации ---
    folder_path: str = Form(None), # Используем None как значение по умолчанию
    collection_name: str = Form(None),
    hf_model: str = Form(None),
    chunk_size: str = Form(None), # Принимаем как строку, проверим и преобразуем позже
    embedding_batch_size: str = Form(None),
    indexing_batch_size: str = Form(None),
    chunk_overlap: str = Form(None),
    use_dense: bool = Form(False), # Checkbox
    device: str = Form(None),
    # --- Поля MinerU ---
    mineru_input_pdf_dir: str = Form(None),
    mineru_output_md_dir: str = Form(None),
    mineru_enable_formula_parsing: bool = Form(False), # Checkbox
    mineru_enable_table_parsing: bool = Form(False),   # Checkbox
    mineru_model_source: str = Form(None),
    mineru_models_dir: Optional[str] = Form(None),
    mineru_backend: str = Form(None),
    mineru_method: str = Form(None),
    mineru_lang: str = Form(None),
    mineru_sglang_url: Optional[str] = Form(None),
    # --- Дополнительные поля ---
    # Настройки кэширования
    config_cache_ttl: str = Form(None),
    qdrant_client_cache_ttl: str = Form(None),
    collections_cache_ttl: str = Form(None),
    # Настройки GGUF моделей
    gguf_model_n_ctx: str = Form(None),
    # Настройки поиска
    search_default_k: str = Form(None),
    # Настройки подключения к Qdrant
    qdrant_url: str = Form(None),
    qdrant_retry_attempts: str = Form(None),
    qdrant_retry_wait_time: str = Form(None),
    # Настройки индексации документов
    memory_threshold: str = Form(None),
    indexing_default_batch_size: str = Form(None),
    sparse_embedding: str = Form(None),
    # Настройки MinerU (дополнительные)
    mineru_subprocess_timeout: str = Form(None),
    # --- Новое поле для определения действия ---
    action: str = Form(...), # Это поле будет определять, какие настройки сохранять
    username: str = Depends(verify_admin_access),
    config: Config = Depends(get_config)
):
    """Обновляет настройки приложения."""
    
    # Импортируем кэш для сброса при смене модели или устройства
    from core.embeddings import _dense_embedder_cache
    
    # Собираем данные формы в словарь
    form_data = {
        "folder_path": folder_path,
        "collection_name": collection_name,
        "hf_model": hf_model,
        "chunk_size": chunk_size,
        "embedding_batch_size": embedding_batch_size,
        "indexing_batch_size": indexing_batch_size,
        "chunk_overlap": chunk_overlap,
        "use_dense": use_dense,
        "device": device,
        "mineru_input_pdf_dir": mineru_input_pdf_dir,
        "mineru_output_md_dir": mineru_output_md_dir,
        "mineru_enable_formula_parsing": mineru_enable_formula_parsing,
        "mineru_enable_table_parsing": mineru_enable_table_parsing,
        "mineru_model_source": mineru_model_source,
        "mineru_models_dir": mineru_models_dir,
        "mineru_backend": mineru_backend,
        "mineru_method": mineru_method,
        "mineru_lang": mineru_lang,
        "mineru_sglang_url": mineru_sglang_url,
        # Дополнительные поля
        "config_cache_ttl": config_cache_ttl,
        "qdrant_client_cache_ttl": qdrant_client_cache_ttl,
        "collections_cache_ttl": collections_cache_ttl,
        "gguf_model_n_ctx": gguf_model_n_ctx,
        "search_default_k": search_default_k,
        "qdrant_url": qdrant_url,
        "qdrant_retry_attempts": qdrant_retry_attempts,
        "qdrant_retry_wait_time": qdrant_retry_wait_time,
        "memory_threshold": memory_threshold,
        "indexing_default_batch_size": indexing_default_batch_size,
        "sparse_embedding": sparse_embedding,
        "mineru_subprocess_timeout": mineru_subprocess_timeout,
    }
    # Получаем все поля исходной формы (включая чекбоксы, которые могут отсутствовать в аргументах функции)
    try:
        raw_form = await request.form()
    except Exception:
        raw_form = {}

    def form_bool(key: str) -> bool:
        # Checkbox присутствует в форме -> treated as True. Also accept explicit 'True'/'true' values.
        if key in raw_form:
            val = raw_form.get(key)
            if isinstance(val, str):
                return val.lower() in ("true", "1", "on")
            return True
        return False

    # Populate explicit index flags from raw form
    form_data["index_dense"] = form_bool("index_dense")
    form_data["index_bm25"] = form_bool("index_bm25")
    form_data["index_hybrid"] = form_bool("index_hybrid")
    # Also populate legacy flags if present
    form_data["use_dense"] = form_bool("use_dense") or form_data.get("use_dense", False)
    form_data["use_hybrid"] = form_bool("use_hybrid") or form_data.get("use_hybrid", False)
    
    # Handle new indexing_type parameter
    indexing_type = raw_form.get("indexing_type")
    if indexing_type == "dense":
        form_data["index_dense"] = True
        form_data["index_bm25"] = False
        form_data["index_hybrid"] = False
    elif indexing_type == "sparse":
        form_data["index_dense"] = False
        form_data["index_bm25"] = True
        form_data["index_hybrid"] = False
    elif indexing_type == "hybrid":
        form_data["index_dense"] = True
        form_data["index_bm25"] = True
        form_data["index_hybrid"] = True
    
    # --- Обработка настроек индексации ---
    if action == "save_index_settings":
        update_index_settings(form_data, config)
        
    # --- Обработка настроек MinerU ---
    elif action == "save_mineru_settings":
        update_mineru_settings(form_data, config)
        
    # --- Обработка дополнительных настроек ---
    elif action == "save_advanced_settings":
        update_advanced_settings(form_data, config)
        
    save_config(config)
    return RedirectResponse(url="/settings?status=saved", status_code=303)


@app.post("/run-indexing", response_class=RedirectResponse)
async def run_indexing(background_tasks: BackgroundTasks, username: str = Depends(verify_admin_access), config: Config = Depends(get_config), client = Depends(get_client)):
    """Запускает процесс индексации документов."""
    background_tasks.add_task(run_indexing_logic, client=client)
    # success, status = await run_indexing_logic(client=client)
    # if success and "successfully" in status:
    #     await refresh_collections_cache(client=client)  # Обновляем кэш после индексации
    return RedirectResponse(url=f"/settings?status=indexing_started", status_code=303)


from fastapi import APIRouter, Request, Form, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
import logging
from pathlib import Path
import os
import shutil
from typing import Dict, Any

from config.settings import load_config, Config, save_config
from core.indexer import run_indexing_from_config
from core.file_converter import run_pdf_processing_from_config, run_multi_format_processing_from_config
from core.qdrant_collections import recreate_collection_from_config
from core.dependencies import get_config, verify_admin_access
from core.constants import TEMPLATES

logger = logging.getLogger(__name__)

app = APIRouter(tags=["admin"])

# ... existing code ...

# Эндпоинт для обработки PDF через MinerU
@app.post("/process-pdfs", response_class=RedirectResponse)
async def process_pdfs_endpoint(background_tasks: BackgroundTasks, username: str = Depends(verify_admin_access), config: Config = Depends(get_config)):
    """Запускает процесс обработки PDF файлов."""
    background_tasks.add_task(run_pdf_processing_from_config)
    # success, status_msg = await run_pdf_processing_from_config()
    return RedirectResponse(url=f"/settings?msg=processing_started", status_code=303)


# Эндпоинт для обработки файлов различных форматов
@app.post("/process-files", response_class=RedirectResponse)
async def process_files_endpoint(background_tasks: BackgroundTasks, username: str = Depends(verify_admin_access), config: Config = Depends(get_config)):
    """Запускает процесс обработки файлов различных форматов."""
    background_tasks.add_task(run_multi_format_processing_from_config)
    return RedirectResponse(url=f"/settings?msg=processing_started", status_code=303)