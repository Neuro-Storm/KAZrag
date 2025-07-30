
import os
import json
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import torch
import uvicorn
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# --- Кэш для embedder'а ---
_dense_embedder = None
_dense_embedder_model = None
_dense_embedder_device = None

# --- Кэш для списка коллекций ---
_cached_collections = None
_cached_collections_time = 0
COLLECTIONS_CACHE_TTL = 300  # 5 минут

def refresh_collections_cache():
    """Обновляет кэш списка коллекций"""
    global _cached_collections, _cached_collections_time
    client = get_qdrant_client()
    try:
        _cached_collections = [c.name for c in client.get_collections().collections]
        _cached_collections_time = time.time()
    except Exception:
        _cached_collections = []
    return _cached_collections

def get_cached_collections():
    """Возвращает кэшированный список коллекций"""
    global _cached_collections, _cached_collections_time
    if (_cached_collections is None or
        (time.time() - _cached_collections_time) > COLLECTIONS_CACHE_TTL):
        return refresh_collections_cache()
    return _cached_collections

# Импорт функции обработки PDF из отдельного модуля
from pdf_to_md_chunker import process_pdfs_and_chunk

# Конфигурация и глобальные переменные приложения
CONFIG_FILE = Path("config.json")
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Вспомогательные функции для работы с конфигурацией
def load_config() -> Dict[str, Any]:
    if not CONFIG_FILE.exists():
        default_cfg = {
            "folder_path": "./data_to_index",
            "collection_name": "final-dense-collection",
            "current_hf_model": "sentence-transformers/all-MiniLM-L6-v2",
            "hf_model_history": ["sentence-transformers/all-MiniLM-L6-v2"],
            "chunk_size": 500,
            "chunk_overlap": 100,
            "qdrant_url": "http://localhost:6333",
            "device": "auto",
            "use_dense_vectors": True,
            "is_indexed": False,
            "mineru_input_pdf_dir": "./pdfs_to_process",
            "mineru_output_md_dir": "./data_to_index",
            "mineru_enable_formula_parsing": False,
            "mineru_enable_table_parsing": False,
            "mineru_model_source": "huggingface",
            "mineru_models_dir": "",
            "mineru_backend": "pipeline",
            "mineru_method": "auto",
            "mineru_lang": "east_slavic",
            "mineru_sglang_url": "",
        }
        save_config(default_cfg)
        return default_cfg
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(config: Dict[str, Any]):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

def get_device(config_device: str) -> str:
    """Определяет устройство для индексации из конфига."""
    if config_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    if config_device == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_search_device(search_device_param: str) -> str:
    """Определяет устройство для поиска из параметра формы."""
    if search_device_param == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=load_config()["qdrant_url"])

# --- Кэширование HuggingFaceEmbeddings ---
def get_dense_embedder(config, device=None):
    global _dense_embedder, _dense_embedder_model, _dense_embedder_device
    model_name = config["current_hf_model"]
    if device is None:
        device = get_device(config["device"])
    # Если модель или устройство изменились, сбрасываем кэш
    if (
        _dense_embedder is None or
        _dense_embedder_model != model_name or
        _dense_embedder_device != device
    ):
        _dense_embedder = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
        _dense_embedder_model = model_name
        _dense_embedder_device = device
    return _dense_embedder

# Основная логика индексации документов
def run_indexing_logic():
    config = load_config()
    if not config.get("use_dense_vectors", False):
        return False, "no_index_type"
    
    folder_path = Path(config["folder_path"])
    folder_path_resolved = folder_path.resolve() # Для получения относительных путей
    if not folder_path.is_dir():
        return False, "folder_not_found"
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"], chunk_overlap=config["chunk_overlap"]
    )
    docs = []
    for filepath in folder_path.rglob("*.txt"):
        try:
            loaded_docs = TextLoader(str(filepath), encoding="utf-8").load()
            chunks = text_splitter.split_documents(loaded_docs)
            try:
                # Получение относительного пути файла от корневой папки
                abs_filepath = filepath.resolve()
                relative_source_path = abs_filepath.relative_to(folder_path_resolved)
                source_str = str(relative_source_path)
            except ValueError:
                # Если файл не в корневой папке, используем полный путь
                print(f"Предупреждение: Файл {filepath} не находится внутри {folder_path_resolved}. Используется полный путь.")
                source_str = str(filepath)

            # ---
            docs.extend(chunks)
        except Exception as e:
            print(f"Ошибка при обработке файла {filepath}: {e}")
            continue
    
    # Добавляем обработку .md файлов (исправлено: всегда используем абсолютные пути)
    for filepath in folder_path.rglob("*.md"):
        try:
            loaded_docs = TextLoader(str(filepath), encoding="utf-8").load()
            chunks = text_splitter.split_documents(loaded_docs)
            abs_filepath = filepath.resolve()
            try:
                relative_source_path = abs_filepath.relative_to(folder_path_resolved)
            except ValueError:
                print(f"Предупреждение: Файл {filepath} не находится внутри {folder_path_resolved}. Используется полный путь.")
                relative_source_path = abs_filepath
            for chunk in chunks:
                chunk.metadata["source"] = str(relative_source_path)
            docs.extend(chunks)
        except Exception as e:
            print(f"Ошибка при обработке файла {filepath}: {e}")
            continue

    # Проверка наличия документов для индексации
    if not docs:
        config["is_indexed"] = False
        save_config(config)
        # ИЗМЕНЕНО: Возвращаем сообщение, которое не считается ошибкой, если файлов просто нет
        # return True, "indexed_successfully_no_docs" 
        # Или, чтобы совсем убрать сообщение, можно вернуть успех:
        return True, "indexed_successfully" 
    try:
        device = get_device(config['device'])
        dense_embedder = get_dense_embedder(config, device)
        QdrantVectorStore.from_documents(
            documents=docs,
            url=config["qdrant_url"],
            collection_name=config["collection_name"],
            embedding=dense_embedder,
            force_recreate=True,
            vector_name="dense_vector",
            batch_size=64,
        )
    except Exception as e:
        print(f"Ошибка во время индексации: {e}")
        print(traceback.format_exc())
        return False, f"indexing_error: {e}"
    config["is_indexed"] = True
    save_config(config)
    return True, "indexed_successfully"

# Эндпоинты FastAPI для веб-интерфейса
@app.get("/", response_class=HTMLResponse)
async def get_search_page(request: Request):
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
    client = get_qdrant_client()
    config = load_config()
    results: List[Tuple[Any, float]] = []
    try:
        if config.get("use_dense_vectors"):
            device = get_search_device(search_device)
            dense_embedder = get_dense_embedder(config, device)
            qdrant_store = QdrantVectorStore(
                client=client,
                collection_name=collection,
                embedding=dense_embedder,
                vector_name="dense_vector",
            )
            results = qdrant_store.similarity_search_with_score(query=query, k=k)
    except Exception as e:
        print(f"Ошибка во время поиска: {e}")
        print(traceback.format_exc())
        results = []
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

@app.post("/settings/delete-collection", response_class=RedirectResponse)
async def delete_collection(request: Request, collection_name: str = Form(...)):
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

@app.get("/settings", response_class=HTMLResponse)
async def get_settings_page(request: Request, status: str = None):
    config = load_config()
    collections = get_cached_collections()
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "config": config,
        "collections": collections,
        "delete_status": status
    })

# Обновление настроек приложения с разделением на типы (индексация/MinerU)
@app.post("/update-settings", response_class=RedirectResponse)
async def update_settings(
    request: Request,
    # --- Поля индексации ---
    folder_path: str = Form(None), # Используем None как значение по умолчанию
    collection_name: str = Form(None),
    hf_model: str = Form(None),
    chunk_size: str = Form(None), # Принимаем как строку, проверим и преобразуем позже
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
    config = load_config()
    
    # --- Обработка настроек индексации ---
    if action == "save_index_settings":
        updates = {}
        global _dense_embedder, _dense_embedder_model, _dense_embedder_device
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
        # Сброс кэша embedder'а при смене модели или устройства
        if model_changed or device_changed:
            _dense_embedder = None
            _dense_embedder_model = None
            _dense_embedder_device = None
        
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
    success, status = run_indexing_logic()
    if success and "successfully" in status:
        refresh_collections_cache()  # Обновляем кэш после индексации
    return RedirectResponse(url=f"/settings?status={status}", status_code=303)

# Эндпоинт для обработки PDF через MinerU
@app.post("/process-pdfs", response_class=RedirectResponse)
async def process_pdfs_endpoint():
    config = load_config()
    try:
        # Вызов функции обработки
        process_pdfs_and_chunk(
            input_pdf_dir=config["mineru_input_pdf_dir"],
            output_md_dir=config["mineru_output_md_dir"],
            enable_formula_parsing=config["mineru_enable_formula_parsing"],
            enable_table_parsing=config["mineru_enable_table_parsing"],
            model_source=config["mineru_model_source"],
            models_dir=config["mineru_models_dir"] if config["mineru_models_dir"] else None,
            backend=config["mineru_backend"],
            method=config["mineru_method"],
            lang=config["mineru_lang"],
            sglang_url=config["mineru_sglang_url"] if config["mineru_sglang_url"] else None,
            device=get_device(config['device']) # Передаем устройство из настроек индексации
        )
        status_msg = "pdfs_processed_successfully"
    except Exception as e:
        print(f"Ошибка при обработке PDF: {e}")
        print(traceback.format_exc())
        status_msg = f"pdf_processing_error_{str(e).replace(' ', '_')}"
    return RedirectResponse(url=f"/settings?status={status_msg}", status_code=303)

if __name__ == "__main__":
    load_config()
    uvicorn.run(app, host="127.0.0.1", port=8000)
