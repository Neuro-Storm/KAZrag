"""FastAPI приложение для индексации и конвертации документов."""

import logging
from pathlib import Path

# Qdrant client provided via dependency injection (core.dependencies.get_client)
from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

from config.config_manager import ConfigManager
from config.settings import Config
from core.converting.multi_format_converter import convert_files_to_md
from core.converting.file_tracker import FileTracker
from core.indexing.indexer import run_indexing_logic_sync
from core.qdrant.qdrant_collections import (
    get_cached_collections,
    refresh_collections_cache,
)

from core.utils.dependencies import get_client, get_config
from core.utils.exception_handlers import get_request_id

logger = logging.getLogger(__name__)

# Initialize config manager
config_manager = ConfigManager.get_instance()

# Словарь соответствий статусов и их сообщений
STATUS_MESSAGES = {
    'indexed_successfully': 'Индексация успешно завершена!',
    'indexed_successfully_no_docs': 'Индексация завершена. Не найдено .txt/.md файлов для обработки.',
    'indexing_started': 'Индексация запущена в фоновом режиме.',
    'processing_started': 'Обработка запущена в фоновом режиме.',
    'pdfs_processed_successfully': 'PDF файлы успешно обработаны!',
    'files_processed_successfully': 'Файлы успешно обработаны!',
    'pdf_processing_error': 'Ошибка при обработке PDF файлов',
    'file_processing_error': 'Ошибка при обработке файлов',
    'indexing_error': 'Ошибка индексации документов',
}

# Словарь соответствий статусов и их типов (success, error, info)
STATUS_TYPES = {
    'indexed_successfully': 'success',
    'indexed_successfully_no_docs': 'success',
    'indexing_started': 'info',
    'processing_started': 'info',
    'pdfs_processed_successfully': 'success',
    'files_processed_successfully': 'success',
    'pdf_processing_error': 'error',
    'file_processing_error': 'error',
    'indexing_error': 'error',
}

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


def get_status_message(status: str) -> str:
    """Получить текстовое сообщение по коду статуса."""
    if status in STATUS_MESSAGES:
        return STATUS_MESSAGES[status]
    
    # Обработка специальных случаев
    if 'error' in status or 'fail' in status:
        return f"Ошибка. Проверьте логи терминала. Сообщение: {status}"
    
    # По умолчанию возвращаем сам статус
    return status


def get_status_type(status: str) -> str:
    """Получить тип статуса (success, error, info)."""
    # Проверяем точные соответствия
    if status in STATUS_TYPES:
        return STATUS_TYPES[status]
    
    # По умолчанию определяем по содержимому
    if 'success' in status or 'indexed' in status or 'processed' in status:
        return 'success'
    elif 'error' in status or 'fail' in status:
        return 'error'
    else:
        return 'info'


# Настройка HTTP Basic Authentication
security = HTTPBasic()


async def verify_admin_access_from_form(request: Request, credentials: HTTPBasicCredentials = Depends(security)):
    """Проверяет учетные данные для доступа к админке."""
    request_id = get_request_id(request)
    
    # Получаем API ключ из переменной окружения
    import os
    ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
    
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


@app.get("/", response_class=HTMLResponse)
async def get_indexing_page(
    request: Request,
    status: str = None,
    username: str = Depends(verify_admin_access_from_form),
    config: Config = Depends(get_config),
):
    """Отображает страницу индексации."""
    request_id = get_request_id(request)
    logger.debug(f"[{request_id}] Loading indexing page for user: {username}")
    
    # Получаем клиент Qdrant с обработкой возможных ошибок
    collections = []
    try:
        from core.qdrant.qdrant_client import get_qdrant_client
        from core.utils.collection_manager import CollectionManager
        client = get_qdrant_client(config)
        
        try:
            # Используем CollectionManager для получения коллекций с информацией
            collection_manager = CollectionManager.get_instance()
            collections_dict = collection_manager.get_collections(client)
            
            # Преобразуем в формат, ожидаемый шаблоном
            collections = []
            for name, collection_info in collections_dict.items():
                # collection_info должен содержать уже всю информацию о коллекции
                points_count = getattr(collection_info, 'points_count', 0) if collection_info else 0
                collections.append({
                    "name": name,
                    "points_count": points_count
                })
        except Exception as collection_error:
            logger.warning(f"[{request_id}] Error getting collections from cache: {collection_error}")
            
    except Exception as client_error:
        logger.warning(f"[{request_id}] Error getting Qdrant client: {client_error}")
    
    try:
        # Формируем сообщение статуса
        status_message = None
        status_type = None
        if status:
            status_message = get_status_message(status)
            status_type = get_status_type(status)
        
        logger.info(f"[{request_id}] Indexing page loaded successfully")
        return get_templates().TemplateResponse("indexing.html", {
            "request": request,
            "config": config,
            "collections": collections,
            "status": status,
            "status_message": status_message,
            "status_type": status_type,
        })
    except Exception as e:
        logger.exception(f"[{request_id}] Error loading indexing page: {str(e)}")
        # Return error page
        return get_templates().TemplateResponse("indexing.html", {
            "request": request,
            "config": config,
            "collections": collections,
            "status": status,
            "status_message": f"Ошибка загрузки страницы индексации: {str(e)}" if status else None,
            "status_type": "error" if status else None,
            "error": "Ошибка загрузки страницы индексации"
        })


@app.post("/run-indexing", response_class=JSONResponse)
async def run_indexing(
    request: Request,
    background_tasks: BackgroundTasks,
    username: str = Depends(verify_admin_access_from_form),
    config: Config = Depends(get_config),
    client = Depends(get_client)
):
    """Запускает процесс индексации документов."""
    request_id = get_request_id(request)
    logger.info(f"[{request_id}] Run indexing request by user: {username}")
    
    try:
        # Use the synchronous wrapper for the indexing logic
        background_tasks.add_task(run_indexing_logic_sync, client=client)
        logger.info(f"[{request_id}] Indexing task scheduled successfully")
        
        return JSONResponse(content={"status": "indexing_started", "message": "Индексация запущена в фоновом режиме"})
        
    except Exception as e:
        logger.exception(f"[{request_id}] Error scheduling indexing task: {str(e)}")
        return JSONResponse(content={"status": "error", "message": "Ошибка запуска индексации"})


@app.post("/process-files", response_class=JSONResponse)
async def process_files_endpoint(
    request: Request,
    background_tasks: BackgroundTasks,
    username: str = Depends(verify_admin_access_from_form),
    config: Config = Depends(get_config)
):
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
        
        return JSONResponse(content={"status": "processing_started", "message": "Обработка файлов запущена"})
        
    except Exception as e:
        logger.exception(f"[{request_id}] Error scheduling file processing task: {str(e)}")
        return JSONResponse(content={"status": "error", "message": "Ошибка запуска обработки файлов"})


@app.post("/update-paths", response_class=JSONResponse)
async def update_paths(
    request: Request,
    input_pdf_dir: str = Form(...),
    output_md_dir: str = Form(...),
    indexing_folder: str = Form(...),
    username: str = Depends(verify_admin_access_from_form),
    config: Config = Depends(get_config)
):
    """Обновляет пути для конвертации и индексации."""
    request_id = get_request_id(request)
    logger.info(f"[{request_id}] Update paths request by user: {username}")
    
    try:
        # Проверяем существование директорий
        if input_pdf_dir and not Path(input_pdf_dir).exists():
            return JSONResponse(
                content={"status": "error", "message": f"Входная директория не существует: {input_pdf_dir}"},
                status_code=400
            )
        
        if output_md_dir and not Path(output_md_dir).exists():
            return JSONResponse(
                content={"status": "error", "message": f"Выходная директория не существует: {output_md_dir}"},
                status_code=400
            )
        
        if indexing_folder and not Path(indexing_folder).exists():
            return JSONResponse(
                content={"status": "error", "message": f"Директория индексации не существует: {indexing_folder}"},
                status_code=400
            )
        
        # Обновляем настройки в конфигурации
        config.mineru_input_pdf_dir = input_pdf_dir
        config.mineru_output_md_dir = output_md_dir
        config.folder_path = indexing_folder
        
        # Сохраняем конфигурацию
        config_manager.save(config)
        
        logger.info(f"[{request_id}] Paths updated successfully:\n  Input PDF dir: {input_pdf_dir}\n  Output MD dir: {output_md_dir}\n  Indexing folder: {indexing_folder}")
        
        return JSONResponse(content={
            "status": "success", 
            "message": "Пути успешно обновлены",
            "paths": {
                "input_pdf_dir": input_pdf_dir,
                "output_md_dir": output_md_dir,
                "indexing_folder": indexing_folder
            }
        })
        
    except Exception as e:
        logger.exception(f"[{request_id}] Error updating paths: {str(e)}")
        return JSONResponse(
            content={"status": "error", "message": f"Ошибка при обновлении путей: {str(e)}"},
            status_code=500
        )


@app.get("/progress")
async def get_conversion_progress():
    """Возвращает статус прогресса конвертации."""
    tracker = FileTracker()
    tracking_data = tracker.get_all_files_status()
    
    total = len(tracking_data)
    converted = len([f for f, d in tracking_data.items() if d["status"] == "converted"])
    errors = len([f for f, d in tracking_data.items() if d["status"] == "error"])
    in_progress = len([f for f, d in tracking_data.items() if d["status"] == "in_progress"])
    pending = total - converted - errors - in_progress
    
    return {
        "total": total,
        "converted": converted,
        "errors": errors,
        "in_progress": in_progress,
        "pending": pending,
        "files": tracking_data
    }


@app.post("/stop-conversion")
async def stop_conversion():
    """Останавливает процесс конвертации."""
    try:
        from core.converting.multi_format_converter import MultiFormatConverter
        converter = MultiFormatConverter()
        converter.stop_conversion()

        logger.info("Конвертация остановлена по запросу пользователя")
        return {"status": "stopped", "message": "Конвертация остановлена"}
    except Exception as e:
        logger.error(f"Ошибка остановки конвертации: {e}")
        return {"status": "error", "message": f"Ошибка остановки: {str(e)}"}


@app.get("/indexing-progress")
async def get_indexing_progress():
    """Возвращает статус прогресса индексации."""
    try:
        from core.indexing.indexing_tracker import IndexingTracker
        tracker = IndexingTracker()
        progress_data = tracker.get_session_progress()

        # Добавляем список всех файлов из директории индексации, если нет активной сессии
        if progress_data.get("status") == "no_session":
            config = config_manager.get()
            indexing_folder = Path(config.folder_path)

            if indexing_folder.exists():
                all_files = []
                for ext in ['.txt', '.md']:
                    all_files.extend(indexing_folder.rglob(f"*{ext}"))
                    all_files.extend(indexing_folder.rglob(f"*{ext.upper()}"))

                files_dict = {}
                for file_path in all_files:
                    if file_path.is_file():
                        str_path = str(file_path.resolve())
                        # Проверяем статус файла из всех сессий
                        status = tracker.get_file_status_from_any_session(file_path)

                        files_dict[str_path] = {
                            "status": status,
                            "updated_at": None,
                            "error_message": None
                        }

                progress_data["files"] = files_dict
                progress_data["total_files"] = len(files_dict)
                progress_data["indexed_files"] = len([f for f in files_dict.values() if f["status"] == "indexed"])
                progress_data["error_files"] = len([f for f in files_dict.values() if f["status"] == "error"])
                progress_data["processed_files"] = len([f for f in files_dict.values() if f["status"] != "pending"])
                progress_data["pending_files"] = len([f for f in files_dict.values() if f["status"] == "pending"])

        # Никакого логирования для API endpoint (чтобы не засорять логи)

        return progress_data
    except Exception as e:
        logger.error(f"Ошибка получения прогресса индексации: {e}")
        return {
            "status": "error",
            "message": f"Ошибка получения прогресса: {str(e)}"
        }


@app.post("/stop-indexing")
async def stop_indexing():
    """Останавливает процесс индексации."""
    try:
        from core.indexing.indexing_tracker import IndexingTracker
        tracker = IndexingTracker()
        tracker.stop_indexing()

        logger.info("Индексация остановлена по запросу пользователя")
        return {"status": "stopped", "message": "Индексация остановлена"}
    except Exception as e:
        logger.error(f"Ошибка остановки индексации: {e}")
        return {"status": "error", "message": f"Ошибка остановки: {str(e)}"}


@app.post("/reset-indexing-tracking")
async def reset_indexing_tracking():
    """Сбрасывает все данные отслеживания индексации."""
    try:
        from core.indexing.indexing_tracker import IndexingTracker
        tracker = IndexingTracker()
        tracker.reset_all()

        logger.info("Данные отслеживания индексации сброшены")
        return {"status": "reset", "message": "Данные отслеживания сброшены"}
    except Exception as e:
        logger.error(f"Ошибка сброса данных отслеживания: {e}")
        return {"status": "error", "message": f"Ошибка сброса: {str(e)}"}


@app.post("/delete-collection", response_class=JSONResponse)
async def delete_collection(
    request: Request,
    username: str = Depends(verify_admin_access_from_form),
    config: Config = Depends(get_config),
    client = Depends(get_client)
):
    """Удаляет указанную коллекцию из Qdrant."""
    request_id = get_request_id(request)
    logger.info(f"[{request_id}] Delete collection request by user: {username}")
    
    try:
        # Получаем данные из тела запроса
        body = await request.json()
        collection_name = body.get("collection_name")
        
        if not collection_name:
            return JSONResponse(
                content={"status": "error", "message": "Не указано имя коллекции"},
                status_code=400
            )
        
        logger.info(f"[{request_id}] Attempting to delete collection: {collection_name}")
        
        # Удаляем коллекцию
        client.delete_collection(collection_name)
        
        # Обновляем кэш коллекций
        from core.qdrant.qdrant_collections import refresh_collections_cache
        await refresh_collections_cache(client=client)
        
        logger.info(f"[{request_id}] Collection '{collection_name}' deleted successfully")
        
        return JSONResponse(content={"status": "deleted", "message": f"Коллекция {collection_name} успешно удалена"})
        
    except Exception as e:
        logger.exception(f"[{request_id}] Error deleting collection '{collection_name}': {str(e)}")
        return JSONResponse(
            content={"status": "error", "message": f"Ошибка при удалении коллекции: {str(e)}"},
            status_code=500
        )