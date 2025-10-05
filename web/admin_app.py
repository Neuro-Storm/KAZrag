"""FastAPI приложение для настроек и управления системой."""

import logging
import os
import time
from pathlib import Path

# Qdrant client provided via dependency injection (core.dependencies.get_client)
from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

from config.config_manager import ConfigManager
from config.settings import Config
from core.converting.multi_format_converter import convert_files_to_md
from core.indexing.indexer import run_indexing_logic
from core.qdrant.qdrant_collections import (
    get_cached_collections,
    refresh_collections_cache,
)

# Initialize config manager
config_manager = ConfigManager.get_instance()

# Словарь соответствий статусов и их сообщений
STATUS_MESSAGES = {
    'saved': 'Настройки сохранены.',
    'indexed_successfully': 'Индексация успешно завершена!',
    'indexed_successfully_no_docs': 'Индексация завершена. Не найдено .txt/.md файлов для обработки.',
    'indexing_started': 'Индексация запущена в фоновом режиме.',
    'processing_started': 'Обработка запущена в фоновом режиме.',
    'error_no_collection_selected': 'Ошибка: Не выбрана коллекция для удаления.',
    'pdfs_processed_successfully': 'PDF файлы успешно обработаны!',
    'files_processed_successfully': 'Файлы успешно обработаны!',
}

# Словарь соответствий статусов и их типов (success, error, info)
STATUS_TYPES = {
    'saved': 'success',
    'indexed_successfully': 'success',
    'indexed_successfully_no_docs': 'success',
    'indexing_started': 'info',
    'processing_started': 'info',
    'pdfs_processed_successfully': 'success',
    'files_processed_successfully': 'success',
    'error_no_collection_selected': 'error',
    'no_index_type': 'error',
    'delete_error': 'error',
    'pdf_processing_error': 'error',
    'file_processing_error': 'error',
    'save_error': 'error',
    'indexing_error': 'error',
}
from core.utils.dependencies import get_client, get_config
from core.utils.exception_handlers import get_request_id

logger = logging.getLogger(__name__)

# Rate limiting storage (in production, use Redis or similar)
_rate_limit_storage = {}

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


# Настройка HTTP Basic Authentication
security = HTTPBasic()

def get_status_message(status: str) -> str:
    """Получить текстовое сообщение по коду статуса."""
    if status in STATUS_MESSAGES:
        return STATUS_MESSAGES[status]
    
    # Обработка специальных случаев
    if 'error' in status or 'fail' in status or 'no_index_type' in status:
        return f"Ошибка индексации. Проверьте логи терминала. Сообщение: {status}"
    
    if 'deleted_' in status and 'delete_error' not in status:
        return "Коллекция успешно удалена."
    
    if 'delete_error' in status:
        error_detail = status.split('delete_error_')[-1].replace('_', ' ')
        return f"Ошибка при удалении коллекции: {error_detail}"
    
    if 'pdf_processing_error' in status:
        return f"Ошибка при обработке PDF. Проверьте логи терминала. Сообщение: {status}"
    
    if 'file_processing_error' in status:
        return f"Ошибка при обработке файлов. Проверьте логи терминала. Сообщение: {status}"
    
    # По умолчанию возвращаем сам статус
    return status


def get_status_type(status: str) -> str:
    """Получить тип статуса (success, error, info)."""
    # Проверяем точные соответствия
    if status in STATUS_TYPES:
        return STATUS_TYPES[status]
    
    # Проверяем частичные соответствия
    for key, value in STATUS_TYPES.items():
        if key in status:
            return value
    
    # По умолчанию определяем по содержимому
    if 'success' in status or 'saved' in status or 'indexed' in status or 'deleted_' in status or 'pdfs_processed' in status or 'files_processed' in status:
        return 'success'
    elif 'error' in status or 'fail' in status or 'no_index_type' in status or 'delete_error' in status or 'pdf_processing_error' in status or 'file_processing_error' in status:
        return 'error'
    else:
        return 'info'


def _check_rate_limit(ip: str, max_requests: int = 10, window: int = 60) -> bool:
    """
    Проверяет, не превышен ли лимит запросов для IP адреса.
    
    Args:
        ip: IP адрес клиента
        max_requests: Максимальное количество запросов за окно
        window: Окно времени в секундах
        
    Returns:
        bool: True если лимит не превышен, False если превышен
    """
    current_time = time.time()
    if ip not in _rate_limit_storage:
        _rate_limit_storage[ip] = []
    
    # Удаляем старые записи
    _rate_limit_storage[ip] = [
        timestamp for timestamp in _rate_limit_storage[ip]
        if current_time - timestamp < window
    ]
    
    # Проверяем лимит
    if len(_rate_limit_storage[ip]) >= max_requests:
        return False
    
    # Добавляем текущий запрос
    _rate_limit_storage[ip].append(current_time)
    return True


def update_index_settings(form_data: dict, config: Config):
    """Обновляет настройки индексации."""
    model_changed = False
    device_changed = False
    
    # Определяем функцию для работы с чекбоксами
    def form_bool(key: str) -> bool:
        # Checkbox присутствует в форме -> treated as True. Also accept explicit 'True'/'true' values.
        if key in form_data:
            val = form_data.get(key)
            if isinstance(val, str):
                return val.lower() in ("true", "1", "on")
            return True
        return False
    
    # Обновляем только те поля, которые переданы в форме
    if "folder_path" in form_data and form_data["folder_path"] is not None:
        folder_path = form_data["folder_path"].strip()
        if not folder_path:
            raise HTTPException(400, detail="Путь к папке не может быть пустым")
        config.folder_path = folder_path
    
    if "collection_name" in form_data and form_data["collection_name"] is not None:
        collection_name = form_data["collection_name"].strip()
        if not collection_name:
            raise HTTPException(400, detail="Имя коллекции не может быть пустым")
        config.collection_name = collection_name
    
    if "hf_model" in form_data and form_data["hf_model"] is not None:
        hf_model = form_data["hf_model"].strip()
        if not hf_model:
            raise HTTPException(400, detail="Модель не может быть пустой")
        config.current_hf_model = hf_model
    
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
    
    # Новые параметры для чанкинга
    if form_data.get("chunking_strategy") is not None:
        config.chunking_strategy = form_data["chunking_strategy"]
    if form_data.get("paragraphs_per_chunk") is not None:
        try:
            config.paragraphs_per_chunk = int(form_data["paragraphs_per_chunk"])
        except ValueError:
            raise HTTPException(400, detail="Неверное количество абзацев в чанке")
    if form_data.get("paragraph_overlap") is not None:
        try:
            config.paragraph_overlap = int(form_data["paragraph_overlap"])
        except ValueError:
            raise HTTPException(400, detail="Неверное перекрытие абзацев")
    if form_data.get("sentences_per_chunk") is not None:
        try:
            config.sentences_per_chunk = int(form_data["sentences_per_chunk"])
        except ValueError:
            raise HTTPException(400, detail="Неверное количество предложений в чанке")
    if form_data.get("sentence_overlap") is not None:
        try:
            config.sentence_overlap = int(form_data["sentence_overlap"])
        except ValueError:
            raise HTTPException(400, detail="Неверное перекрытие предложений")
    
    # Параметры для многоуровневого чанкинга
    config.use_multilevel_chunking = form_bool("use_multilevel_chunking")
    if form_data.get("multilevel_macro_strategy") is not None:
        config.multilevel_macro_strategy = form_data["multilevel_macro_strategy"]
    if form_data.get("multilevel_macro_chunk_size") is not None:
        try:
            config.multilevel_macro_chunk_size = int(form_data["multilevel_macro_chunk_size"])
        except ValueError:
            raise HTTPException(400, detail="Неверный размер макро-чанка")
    if form_data.get("multilevel_macro_chunk_overlap") is not None:
        try:
            config.multilevel_macro_chunk_overlap = int(form_data["multilevel_macro_chunk_overlap"])
        except ValueError:
            raise HTTPException(400, detail="Неверное перекрытие макро-чанков")
    if form_data.get("multilevel_macro_paragraphs_per_chunk") is not None:
        try:
            config.multilevel_macro_paragraphs_per_chunk = int(form_data["multilevel_macro_paragraphs_per_chunk"])
        except ValueError:
            raise HTTPException(400, detail="Неверное количество абзацев в макро-чанке")
    if form_data.get("multilevel_macro_paragraph_overlap") is not None:
        try:
            config.multilevel_macro_paragraph_overlap = int(form_data["multilevel_macro_paragraph_overlap"])
        except ValueError:
            raise HTTPException(400, detail="Неверное перекрытие абзацев в макро-чанках")
    if form_data.get("multilevel_macro_sentences_per_chunk") is not None:
        try:
            config.multilevel_macro_sentences_per_chunk = int(form_data["multilevel_macro_sentences_per_chunk"])
        except ValueError:
            raise HTTPException(400, detail="Неверное количество предложений в макро-чанке")
    if form_data.get("multilevel_macro_sentence_overlap") is not None:
        try:
            config.multilevel_macro_sentence_overlap = int(form_data["multilevel_macro_sentence_overlap"])
        except ValueError:
            raise HTTPException(400, detail="Неверное перекрытие предложений в макро-чанках")
    if form_data.get("multilevel_micro_strategy") is not None:
        config.multilevel_micro_strategy = form_data["multilevel_micro_strategy"]
    if form_data.get("multilevel_micro_chunk_size") is not None:
        try:
            config.multilevel_micro_chunk_size = int(form_data["multilevel_micro_chunk_size"])
        except ValueError:
            raise HTTPException(400, detail="Неверный размер микро-чанка")
    if form_data.get("multilevel_micro_chunk_overlap") is not None:
        try:
            config.multilevel_micro_chunk_overlap = int(form_data["multilevel_micro_chunk_overlap"])
        except ValueError:
            raise HTTPException(400, detail="Неверное перекрытие микро-чанков")
    if form_data.get("multilevel_micro_paragraphs_per_chunk") is not None:
        try:
            config.multilevel_micro_paragraphs_per_chunk = int(form_data["multilevel_micro_paragraphs_per_chunk"])
        except ValueError:
            raise HTTPException(400, detail="Неверное количество абзацев в микро-чанке")
    if form_data.get("multilevel_micro_paragraph_overlap") is not None:
        try:
            config.multilevel_micro_paragraph_overlap = int(form_data["multilevel_micro_paragraph_overlap"])
        except ValueError:
            raise HTTPException(400, detail="Неверное перекрытие абзацев в микро-чанках")
    if form_data.get("multilevel_micro_sentences_per_chunk") is not None:
        try:
            config.multilevel_micro_sentences_per_chunk = int(form_data["multilevel_micro_sentences_per_chunk"])
        except ValueError:
            raise HTTPException(400, detail="Неверное количество предложений в микро-чанке")
    if form_data.get("multilevel_micro_sentence_overlap") is not None:
        try:
            config.multilevel_micro_sentence_overlap = int(form_data["multilevel_micro_sentence_overlap"])
        except ValueError:
            raise HTTPException(400, detail="Неверное перекрытие предложений в микро-чанках")
    
    # New explicit flags for index type (use_dense_vectors and use_hybrid were removed)
    config.index_dense = bool(form_data.get("index_dense", form_data.get("use_dense", False)))
    config.index_bm25 = bool(form_data.get("index_bm25", False))
    config.index_hybrid = bool(form_data.get("index_hybrid", form_data.get("use_hybrid", False)))
    # keep legacy flag for backward compatibility (but it's no longer in the model)
    # config.use_hybrid = form_data.get("use_hybrid", False)  # This field was removed
    
    if form_data.get("device") is not None:
        config.device = form_data["device"]
        if form_data["device"] != config.device:
            device_changed = True
    
    # config.is_indexed = False # Сбрасываем флаг индексации (поле было удалено)

    if form_data.get("hf_model") is not None and form_data["hf_model"] not in config.hf_model_history:
        config.hf_model_history.append(form_data["hf_model"])

    # Сброс кэшa embedder'а при смене модели или устройства
    if model_changed or device_changed:
        from core.embeddings import clear_embedder_cache
        clear_embedder_cache()
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
    # Обновляем модель для плотных векторов, если она передана
    if "hf_model" in form_data and form_data["hf_model"] is not None:
        hf_model = form_data["hf_model"].strip()
        if hf_model:
            config.current_hf_model = hf_model
            # Добавляем модель в историю, если её там ещё нет
            if hf_model not in config.hf_model_history:
                config.hf_model_history.append(hf_model)
    
    # Sparse embedding model
    if form_data.get("sparse_embedding") is not None:
        sparse_embedding = form_data["sparse_embedding"].strip()
        if sparse_embedding:
            config.sparse_embedding = sparse_embedding
    
    # Настройки BM25
    config.use_bm25 = "use_bm25" in form_data
    if form_data.get("sparse_vector_name") is not None:
        config.sparse_vector_name = form_data["sparse_vector_name"].strip()
    if form_data.get("bm25_tokenizer") is not None:
        config.bm25_tokenizer = form_data["bm25_tokenizer"].strip()
    if form_data.get("bm25_min_token_len") is not None:
        try:
            config.bm25_min_token_len = int(form_data["bm25_min_token_len"])
        except ValueError:
            raise HTTPException(400, detail="Неверная минимальная длина токена для BM25")
    
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
    
    # Настройки reranker
    config.reranker_enabled = "reranker_enabled" in form_data
    if form_data.get("reranker_model") is not None:
        config.reranker_model = form_data["reranker_model"].strip()
    if form_data.get("reranker_top_k") is not None:
        try:
            config.reranker_top_k = int(form_data["reranker_top_k"])
        except ValueError:
            raise HTTPException(400, detail="Неверное количество результатов после reranking")
    
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
            raise HTTPException(400, detail="Неверное размер батча для индексации документов по умолчанию")
    if form_data.get("sparse_embedding") is not None:
        config.sparse_embedding = form_data["sparse_embedding"].strip()
    
    # Настройки MinerU (дополнительные)
    if form_data.get("mineru_subprocess_timeout") is not None:
        try:
            config.mineru_subprocess_timeout = int(form_data["mineru_subprocess_timeout"])
        except ValueError:
            raise HTTPException(400, detail="Неверный таймаут subprocess вызова mineru")
    
    # Настройки RAG
    config.rag_enabled = "rag_enabled" in form_data
    if form_data.get("rag_model_path") is not None:
        config.rag_model_path = form_data["rag_model_path"].strip()
    if form_data.get("rag_system_prompt") is not None:
        config.rag_system_prompt = form_data["rag_system_prompt"].strip()
    if form_data.get("rag_top_k") is not None:
        try:
            config.rag_top_k = int(form_data["rag_top_k"])
        except ValueError:
            raise HTTPException(400, detail="Неверное количество топ-K результатов для RAG")
    if form_data.get("rag_max_tokens") is not None:
        try:
            config.rag_max_tokens = int(form_data["rag_max_tokens"])
        except ValueError:
            raise HTTPException(400, detail="Неверное максимальное количество токенов для RAG")
    if form_data.get("rag_temperature") is not None:
        try:
            config.rag_temperature = float(form_data["rag_temperature"])
        except ValueError:
            raise HTTPException(400, detail="Неверная температура генерации для RAG")
    if form_data.get("rag_context_size") is not None:
        try:
            config.rag_context_size = int(form_data["rag_context_size"])
        except ValueError:
            raise HTTPException(400, detail="Неверный размер контекста для RAG")
    if form_data.get("rag_gpu_layers") is not None:
        try:
            config.rag_gpu_layers = int(form_data["rag_gpu_layers"])
        except ValueError:
            raise HTTPException(400, detail="Неверное количество слоев для GPU в RAG")
    if form_data.get("rag_threads") is not None:
        try:
            config.rag_threads = int(form_data["rag_threads"])
        except ValueError:
            raise HTTPException(400, detail="Неверное количество потоков для RAG")
    if form_data.get("rag_batch_size") is not None:
        try:
            config.rag_batch_size = int(form_data["rag_batch_size"])
        except ValueError:
            raise HTTPException(400, detail="Неверное размер батча для RAG")
    if form_data.get("rag_beam_size") is not None:
        try:
            config.rag_beam_size = int(form_data["rag_beam_size"])
        except ValueError:
            raise HTTPException(400, detail="Неверный размер beam для RAG")
    
    # Глобальные настройки поиска
    if form_data.get("search_default_collection") is not None:
        config.search_default_collection = form_data["search_default_collection"].strip()
    if form_data.get("search_default_device") is not None:
        config.search_default_device = form_data["search_default_device"].strip()
    if form_data.get("search_default_type") is not None:
        config.search_default_type = form_data["search_default_type"].strip()
    config.search_default_use_reranker = "search_default_use_reranker" in form_data


async def verify_admin_access_from_form(request: Request, credentials: HTTPBasicCredentials = Depends(security)):
    """Проверяет учетные данные для доступа к админке."""
    request_id = get_request_id(request)
    
    # Получаем API ключ из переменной окружения
    ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
    
    # Проверка rate-limit
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip, max_requests=100, window=60):
        logger.warning(f"[{request_id}] Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=429,
            detail="Превышен лимит запросов. Попробуйте позже.",
        )
    
    # Если API ключ не установлен в .env, разрешаем доступ без аутентификации
    if not ADMIN_API_KEY:
        logger.warning(f"[{request_id}] ADMIN_API_KEY not set, access granted (no authentication required)")
        return "anonymous"
    
    # Проверяем учетные данные
    # В данном случае используем API ключ как пароль, имя пользователя может быть любым
    if credentials.password == ADMIN_API_KEY:
        return credentials.username
    
    # В противном случае, возвращаем ошибку
    logger.warning(f"[{request_id}] Invalid admin credentials provided")
    raise HTTPException(
        status_code=401,
        detail="Неверные учетные данные",
        headers={"WWW-Authenticate": "Basic"},
    )


@app.get("/settings/", response_class=HTMLResponse)
async def get_settings_page(request: Request, status: str = None, username: str = Depends(verify_admin_access_from_form), config: Config = Depends(get_config), client = Depends(get_client)):
    """Отображает страницу настроек."""
    request_id = get_request_id(request)
    logger.debug(f"[{request_id}] Loading settings page for user: {username}")
    
    try:
        logger.debug(f"[{request_id}] Configuration loaded: {config.dict()}")
        collections = get_cached_collections(client=client)
        # Получаем директорию кэша fastembed из переменных окружения
        fastembed_cache_dir = os.environ.get('FASTEMBED_CACHE_DIR')
        
        # Формируем сообщение статуса
        status_message = None
        status_type = None
        if status:
            status_message = get_status_message(status)
            status_type = get_status_type(status)
        
        logger.info(f"[{request_id}] Settings page loaded successfully")
        return get_templates().TemplateResponse("settings.html", {
            "request": request,
            "config": config,
            "collections": collections,
            "status": status,
            "status_message": status_message,
            "status_type": status_type,
            "delete_status": status if status and 'delete' in status else None,
            "fastembed_cache_dir": fastembed_cache_dir
        })
    except Exception as e:
        logger.exception(f"[{request_id}] Error loading settings page: {str(e)}")
        # Return error page
        return get_templates().TemplateResponse("settings.html", {
            "request": request,
            "config": config,
            "collections": [],
            "status": status,
            "status_message": f"Ошибка загрузки страницы настроек: {str(e)}" if status else None,
            "status_type": "error" if status else None,
            "delete_status": status if status and 'delete' in status else None,
            "error": "Ошибка загрузки страницы настроек"
        })


@app.post("/settings/delete-collection", response_class=RedirectResponse)
async def delete_collection(request: Request, collection_name: str = Form(...), username: str = Depends(verify_admin_access_from_form), config: Config = Depends(get_config), client = Depends(get_client)):
    """Удаляет указанную коллекцию из Qdrant."""
    request_id = get_request_id(request)
    logger.info(f"[{request_id}] Delete collection request: {collection_name} by user: {username}")
    
    status_msg = ""
    try:
        if not collection_name:
            status_msg = "error_no_collection_selected"
            logger.warning(f"[{request_id}] No collection name provided for deletion")
        else:
            client.delete_collection(collection_name)
            await refresh_collections_cache(client=client)  # Обновляем кэш после удаления
            status_msg = f"deleted_{collection_name}"
            logger.info(f"[{request_id}] Collection '{collection_name}' deleted successfully")
    except Exception as e:
        logger.exception(f"[{request_id}] Error deleting collection '{collection_name}': {str(e)}")
        status_msg = "delete_error"
    
    # Проверяем, является ли запрос AJAX запросом
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
    
    if is_ajax:
        # Для AJAX запросов возвращаем JSON ответ
        return JSONResponse(content={"status": status_msg})
    else:
        # Для обычных запросов возвращаем редирект
        # При удалении коллекции обычно активна вкладка коллекций
        return RedirectResponse(url=f"/settings?status={status_msg}&tab=collections", status_code=303)


# Обновление настроек приложения с разделением на типы (индексация/MinerU/Дополнительные)
@app.post("/update-settings", response_class=RedirectResponse)
async def update_settings(
    request: Request,
    action: str = Form(...), # Это поле будет определять, какие настройки сохранять
    username: str = Depends(verify_admin_access_from_form),
    config: Config = Depends(get_config)
):
    """Обновляет настройки приложения."""
    request_id = get_request_id(request)
    logger.info(f"[{request_id}] Update settings request: action={action} by user: {username}")
    
    try:
        # Импортируем кэш для сброса при смене модели или устройства
        
        # Получаем все поля формы
        try:
            raw_form = await request.form()
        except Exception:
            raw_form = {}

        # Собираем данные формы в словарь только из фактически переданных полей
        form_data = dict(raw_form)
        def form_bool(key: str) -> bool:
            # Checkbox присутствует в форме -> treated as True. Also accept explicit 'True'/'true' values.
            if key in form_data:
                val = form_data.get(key)
                if isinstance(val, str):
                    return val.lower() in ("true", "1", "on")
                return True
            return False

        # Populate explicit index flags from form data
        form_data["index_dense"] = form_bool("index_dense")
        form_data["index_bm25"] = form_bool("index_bm25")
        form_data["index_hybrid"] = form_bool("index_hybrid")
        # Also populate legacy flags if present
        form_data["use_dense"] = form_bool("use_dense") or form_data.get("use_dense", False)
        form_data["use_hybrid"] = form_bool("use_hybrid") or form_data.get("use_hybrid", False)
        
        # Handle new indexing_type parameter
        indexing_type = form_data.get("indexing_type")
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
            
        config_manager.save(config)
        logger.info(f"[{request_id}] Settings updated successfully: action={action}")
        
        # Проверяем, является ли запрос AJAX запросом
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
        
        if is_ajax:
            # Для AJAX запросов возвращаем JSON ответ
            return JSONResponse(content={"status": "saved"})
        else:
            # Для обычных запросов возвращаем редирект с информацией о активной вкладке
            active_tab = form_data.get("active_tab", "indexing")
            return RedirectResponse(url=f"/settings?status=saved&tab={active_tab}", status_code=303)
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] Error updating settings: {str(e)}")
        # Проверяем, является ли запрос AJAX запросом
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
        
        if is_ajax:
            # Для AJAX запросов возвращаем JSON ответ
            return JSONResponse(content={"status": "error", "message": "Ошибка сохранения настроек"})
        else:
            # Для обычных запросов возвращаем редирект с сообщением об ошибке и информацией о активной вкладке
            active_tab = form_data.get("active_tab", "indexing")
            return RedirectResponse(url=f"/settings?status=save_error&tab={active_tab}", status_code=303)


@app.post("/run-indexing", response_class=RedirectResponse)
async def run_indexing(request: Request, background_tasks: BackgroundTasks, username: str = Depends(verify_admin_access_from_form), config: Config = Depends(get_config), client = Depends(get_client)):
    """Запускает процесс индексации документов."""
    request_id = get_request_id(request)
    logger.info(f"[{request_id}] Run indexing request by user: {username}")
    
    try:
        # Use the synchronous wrapper for the indexing logic
        from core.indexing.indexer import run_indexing_logic_sync
        background_tasks.add_task(run_indexing_logic_sync, client=client)
        logger.info(f"[{request_id}] Indexing task scheduled successfully")
        
        # Проверяем, является ли запрос AJAX запросом
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
        
        if is_ajax:
            # Для AJAX запросов возвращаем JSON ответ
            return JSONResponse(content={"status": "indexing_started"})
        else:
            # Для обычных запросов возвращаем редирект с информацией о активной вкладке
            # При запуске индексации обычно активна вкладка индексации
            return RedirectResponse(url="/settings?status=indexing_started&tab=indexing", status_code=303)
            
    except Exception as e:
        logger.exception(f"[{request_id}] Error scheduling indexing task: {str(e)}")
        # Проверяем, является ли запрос AJAX запросом
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
        
        if is_ajax:
            # Для AJAX запросов возвращаем JSON ответ
            return JSONResponse(content={"status": "error", "message": "Ошибка запуска индексации"})
        else:
            # Для обычных запросов возвращаем редирект с сообщением об ошибке
            # При ошибке индексации обычно активна вкладка индексации
            return RedirectResponse(url="/settings?status=indexing_error&tab=indexing", status_code=303)


# Эндпоинт для обработки файлов различных форматов
@app.post("/process-files", response_class=RedirectResponse)
async def process_files_endpoint(request: Request, background_tasks: BackgroundTasks, username: str = Depends(verify_admin_access_from_form), config: Config = Depends(get_config)):
    """Запускает процесс обработки файлов различных форматов."""
    request_id = get_request_id(request)
    logger.info(f"[{request_id}] Process files request by user: {username}")
    
    try:
        # Запускаем обработку файлов в фоновом режиме
        config = config_manager.get()
        
        # Create a sync wrapper function to run the potentially long-running task
        def run_convert_files_sync():
            convert_files_to_md(
                input_dir=config.mineru_input_pdf_dir,
                output_dir=config.mineru_output_md_dir
            )
        
        background_tasks.add_task(run_convert_files_sync)
        logger.info(f"[{request_id}] Multi-format processing task scheduled successfully")
        
        # Проверяем, является ли запрос AJAX запросом
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
        
        if is_ajax:
            # Для AJAX запросов возвращаем JSON ответ
            return JSONResponse(content={"status": "processing_started"})
        else:
            # Для обычных запросов возвращаем редирект с информацией о активной вкладке
            # При запуске обработки файлов обычно активна вкладка конвертации
            return RedirectResponse(url="/settings?status=processing_started&tab=conversion", status_code=303)
            
    except Exception as e:
        logger.exception(f"[{request_id}] Error scheduling file processing task: {str(e)}")
        # Проверяем, является ли запрос AJAX запросом
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
        
        if is_ajax:
            # Для AJAX запросов возвращаем JSON ответ
            return JSONResponse(content={"status": "error", "message": "Ошибка запуска обработки файлов"})
        else:
            # Для обычных запросов возвращаем редирект с сообщением об ошибке
            # При ошибке обработки файлов обычно активна вкладка конвертации
            return RedirectResponse(url="/settings?status=processing_error&tab=conversion", status_code=303)