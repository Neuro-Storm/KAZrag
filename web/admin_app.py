"""FastAPI приложение для настроек и управления системой."""

import traceback
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from qdrant_client import QdrantClient

from config.settings import load_config, save_config
from core.indexer import run_indexing_logic, get_qdrant_client
from core.file_converter import run_pdf_processing_from_config
from core.qdrant_collections import get_cached_collections, refresh_collections_cache

# Инициализация FastAPI приложения и шаблонов
app = FastAPI()
templates = Jinja2Templates(directory="web/templates")


@app.get("/settings", response_class=HTMLResponse)
async def get_settings_page(request: Request, status: str = None):
    """Отображает страницу настроек."""
    config = load_config()
    collections = get_cached_collections()
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "config": config,
        "collections": collections,
        "delete_status": status
    })


@app.post("/settings/delete-collection", response_class=RedirectResponse)
async def delete_collection(request: Request, collection_name: str = Form(...)):
    """Удаляет указанную коллекцию из Qdrant."""
    client = get_qdrant_client()
    status_msg = ""
    try:
        if not collection_name:
             status_msg = "error_no_collection_selected"
        else:
            client.delete_collection(collection_name)
            refresh_collections_cache()  # Обновляем кэш после удаления
            status_msg = f"deleted_{collection_name}"
    except Exception as e:
        print(f"Ошибка при удалении коллекции '{collection_name}': {e}")
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
    use_dense: str = Form(None), # Checkbox приходит как "True" или None
    device: str = Form(None),
    # --- Поля MinerU ---
    mineru_input_pdf_dir: str = Form(None),
    mineru_output_md_dir: str = Form(None),
    mineru_enable_formula_parsing: str = Form(None), # Checkbox
    mineru_enable_table_parsing: str = Form(None),   # Checkbox
    mineru_model_source: str = Form(None),
    mineru_models_dir: str = Form(""),
    mineru_backend: str = Form(None),
    mineru_method: str = Form(None),
    mineru_lang: str = Form(None),
    mineru_sglang_url: str = Form(""),
    # --- Новое поле для определения действия ---
    action: str = Form(...) # Это поле будет определять, какие настройки сохранять
):
    """Обновляет настройки приложения."""
    config = load_config()
    
    # Импортируем кэш для сброса при смене модели или устройства
    from core.embeddings import _dense_embedder_cache
    
    # --- Обработка настроек индексации ---
    if action == "save_index_settings":
        updates = {}
        model_changed = False
        device_changed = False
        if folder_path is not None: updates["folder_path"] = folder_path
        if collection_name is not None: updates["collection_name"] = collection_name
        if hf_model is not None:
            updates["current_hf_model"] = hf_model
            if hf_model != config.get("current_hf_model"):
                model_changed = True
        if chunk_size is not None:
            try:
                updates["chunk_size"] = int(chunk_size)
            except ValueError:
                pass # Игнорируем некорректные значения
        if chunk_overlap is not None:
            try:
                updates["chunk_overlap"] = int(chunk_overlap)
            except ValueError:
                pass
        if embedding_batch_size is not None:
            try:
                updates["embedding_batch_size"] = int(embedding_batch_size)
            except ValueError:
                pass # Игнорируем некорректные значения
        if indexing_batch_size is not None:
            try:
                updates["indexing_batch_size"] = int(indexing_batch_size)
            except ValueError:
                pass # Игнорируем некорректные значения
        # use_dense приходит как "True" или None
        updates["use_dense_vectors"] = use_dense == "True"
        if device is not None:
            updates["device"] = device
            if device != config.get("device"):
                device_changed = True
        updates["is_indexed"] = False # Сбрасываем флаг индексации

        if hf_model is not None and hf_model not in config["hf_model_history"]:
            config["hf_model_history"].append(hf_model)

        config.update(updates)
        # Сброс кэшa embedder'а при смене модели или устройства
        if model_changed or device_changed:
            from core.embeddings import _dense_embedder_cache
            _dense_embedder_cache.clear()
            # Принудительно очищаем GPU память, если используется CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        
    # --- Обработка настроек MinerU ---
    elif action == "save_mineru_settings":
        updates = {}
        if mineru_input_pdf_dir is not None: updates["mineru_input_pdf_dir"] = mineru_input_pdf_dir
        if mineru_output_md_dir is not None: updates["mineru_output_md_dir"] = mineru_output_md_dir
        # Checkboxes
        updates["mineru_enable_formula_parsing"] = mineru_enable_formula_parsing == "True"
        updates["mineru_enable_table_parsing"] = mineru_enable_table_parsing == "True"
        if mineru_model_source is not None: updates["mineru_model_source"] = mineru_model_source
        if mineru_models_dir is not None: updates["mineru_models_dir"] = mineru_models_dir
        if mineru_backend is not None: updates["mineru_backend"] = mineru_backend
        if mineru_method is not None: updates["mineru_method"] = mineru_method
        if mineru_lang is not None: updates["mineru_lang"] = mineru_lang
        if mineru_sglang_url is not None: updates["mineru_sglang_url"] = mineru_sglang_url
        
        config.update(updates)

    save_config(config)
    return RedirectResponse(url="/settings?status=saved", status_code=303)


@app.post("/run-indexing", response_class=RedirectResponse)
async def run_indexing():
    """Запускает процесс индексации документов."""
    success, status = run_indexing_logic()
    if success and "successfully" in status:
        refresh_collections_cache()  # Обновляем кэш после индексации
    return RedirectResponse(url=f"/settings?status={status}", status_code=303)


# Эндпоинт для обработки PDF через MinerU
@app.post("/process-pdfs", response_class=RedirectResponse)
async def process_pdfs_endpoint():
    """Запускает процесс обработки PDF файлов."""
    success, status_msg = run_pdf_processing_from_config()
    return RedirectResponse(url=f"/settings?status={status_msg}", status_code=303)