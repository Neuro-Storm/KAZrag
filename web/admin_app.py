"""FastAPI приложение для настроек и управления системой."""

import traceback
import os
import logging
from typing import Optional
from fastapi import FastAPI, Form, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
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

async def verify_admin_access(credentials: HTTPBasicCredentials = Depends(security)):
    """Проверяет учетные данные для доступа к админке."""
    # Если API ключ не установлен в .env, разрешаем доступ без аутентификации
    if not ADMIN_API_KEY:
        logger.warning("No API key set for admin access. Admin access is open.")
        return None
    
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
async def get_settings_page(request: Request, status: str = None, username: str = Depends(verify_admin_access), config: Config = Depends(get_config)):
    """Отображает страницу настроек."""
    collections = get_cached_collections()
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "config": config,
        "collections": collections,
        "delete_status": status
    })


@app.post("/settings/delete-collection", response_class=RedirectResponse)
async def delete_collection(request: Request, collection_name: str = Form(...), username: str = Depends(verify_admin_access), config: Config = Depends(get_config), client: QdrantClient = Depends(get_client)):
    """Удаляет указанную коллекцию из Qdrant."""
    status_msg = ""
    try:
        if not collection_name:
             status_msg = "error_no_collection_selected"
        else:
            client.delete_collection(collection_name)
            refresh_collections_cache(client=client)  # Обновляем кэш после удаления
            status_msg = f"deleted_{collection_name}"
    except Exception as e:
        logger.error(f"Ошибка при удалении коллекции '{collection_name}': {e}")
        status_msg = f"delete_error_{str(e).replace(' ', '_')}"
    return RedirectResponse(url=f"/settings?status={status_msg}", status_code=303)


# Обновление настроек приложения с разделением на типы (индексация/MinerU)
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
    # --- Новое поле для определения действия ---
    action: str = Form(...), # Это поле будет определять, какие настройки сохранять
    username: str = Depends(verify_admin_access),
    config: Config = Depends(get_config)
):
    """Обновляет настройки приложения."""
    
    # Импортируем кэш для сброса при смене модели или устройства
    from core.embeddings import _dense_embedder_cache
    
    # --- Обработка настроек индексации ---
    if action == "save_index_settings":
        updates = {}
        model_changed = False
        device_changed = False
        if folder_path is not None: config.folder_path = folder_path
        if collection_name is not None: config.collection_name = collection_name
        if hf_model is not None:
            config.current_hf_model = hf_model
            if hf_model != config.current_hf_model:
                model_changed = True
        if chunk_size is not None:
            try:
                config.chunk_size = int(chunk_size)
            except ValueError:
                # Можно добавить flash сообщение об ошибке
                return RedirectResponse(url="/settings?status=invalid_chunk_size", status_code=303)
        if chunk_overlap is not None:
            try:
                config.chunk_overlap = int(chunk_overlap)
            except ValueError:
                # Можно добавить flash сообщение об ошибке
                return RedirectResponse(url="/settings?status=invalid_chunk_overlap", status_code=303)
        if embedding_batch_size is not None:
            try:
                config.embedding_batch_size = int(embedding_batch_size)
            except ValueError:
                # Можно добавить flash сообщение об ошибке
                return RedirectResponse(url="/settings?status=invalid_embedding_batch_size", status_code=303)
        if indexing_batch_size is not None:
            try:
                config.indexing_batch_size = int(indexing_batch_size)
            except ValueError:
                # Можно добавить flash сообщение об ошибке
                return RedirectResponse(url="/settings?status=invalid_indexing_batch_size", status_code=303)
        # use_dense теперь bool
        config.use_dense_vectors = use_dense
        if device is not None:
            config.device = device
            if device != config.device:
                device_changed = True
        config.is_indexed = False # Сбрасываем флаг индексации

        if hf_model is not None and hf_model not in config.hf_model_history:
            config.hf_model_history.append(hf_model)

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
        
    # --- Обработка настроек MinerU ---
    elif action == "save_mineru_settings":
        if mineru_input_pdf_dir is not None: config.mineru_input_pdf_dir = mineru_input_pdf_dir
        if mineru_output_md_dir is not None: config.mineru_output_md_dir = mineru_output_md_dir
        # Checkboxes теперь bool
        config.mineru_enable_formula_parsing = mineru_enable_formula_parsing
        config.mineru_enable_table_parsing = mineru_enable_table_parsing
        if mineru_model_source is not None: config.mineru_model_source = mineru_model_source
        # Optional поля
        if mineru_models_dir is not None and mineru_models_dir.strip(): 
            config.mineru_models_dir = mineru_models_dir.strip()
        if mineru_backend is not None: config.mineru_backend = mineru_backend
        if mineru_method is not None: config.mineru_method = mineru_method
        if mineru_lang is not None: config.mineru_lang = mineru_lang
        # Optional поля
        if mineru_sglang_url is not None and mineru_sglang_url.strip(): 
            config.mineru_sglang_url = mineru_sglang_url.strip()
        
    save_config(config)
    return RedirectResponse(url="/settings?status=saved", status_code=303)


@app.post("/run-indexing", response_class=RedirectResponse)
async def run_indexing(username: str = Depends(verify_admin_access), config: Config = Depends(get_config), client: QdrantClient = Depends(get_client)):
    """Запускает процесс индексации документов."""
    success, status = run_indexing_logic(client=client)
    if success and "successfully" in status:
        refresh_collections_cache(client=client)  # Обновляем кэш после индексации
    return RedirectResponse(url=f"/settings?status={status}", status_code=303)


# Эндпоинт для обработки PDF через MinerU
@app.post("/process-pdfs", response_class=RedirectResponse)
async def process_pdfs_endpoint(username: str = Depends(verify_admin_access), config: Config = Depends(get_config)):
    """Запускает процесс обработки PDF файлов."""
    success, status_msg = run_pdf_processing_from_config()
    return RedirectResponse(url=f"/settings?status={status_msg}", status_code=303)