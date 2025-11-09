"""Модуль для отслеживания состояния файлов во время индексации."""

import json
from pathlib import Path
from typing import Dict, Any, List
import logging
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

# Глобальное событие для остановки индексации
indexing_stop_event = threading.Event()


class IndexingTracker:
    """Отслеживает состояние файлов для индексации."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path: Path = Path("indexing_tracking.json")):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(IndexingTracker, cls).__new__(cls)
                    cls._instance.db_path = db_path
                    cls._instance.data = {}
                    cls._instance.session_id = None
                    cls._instance.load()
        return cls._instance

    def __init__(self, db_path: Path = Path("indexing_tracking.json")):
        # Инициализация уже произошла в __new__
        pass

    def load(self):
        """Загрузить данные отслеживания из файла."""
        if self.db_path.exists():
            try:
                content = self.db_path.read_text(encoding="utf-8")
                self.data = json.loads(content) if content.strip() else {}
            except Exception as e:
                logger.warning(f"Ошибка чтения {self.db_path}: {e}")
                self.data = {}
        else:
            self.data = {}

    def save(self):
        """Сохранить данные отслеживания в файл."""
        try:
            self.db_path.write_text(
                json.dumps(self.data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"Ошибка сохранения {self.db_path}: {e}")

    def start_new_session(self, collection_name: str = None):
        """Начать новую сессию индексации."""
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data["current_session"] = self.session_id

        if "sessions" not in self.data:
            self.data["sessions"] = {}

        self.data["sessions"][self.session_id] = {
            "started_at": datetime.now().isoformat(),
            "status": "running",
            "total_files": 0,
            "processed_files": 0,
            "indexed_files": 0,
            "error_files": 0,
            "current_file": None,
            "current_stage": None,
            "stop_requested": False,
            "collection_name": collection_name,
            "files": {}
        }
        self.save()
        logger.info(f"Начата новая сессия индексации: {self.session_id} (коллекция: {collection_name})")

    def end_session(self, status: str = "completed"):
        """Завершить текущую сессию индексации."""
        if not self.session_id or self.session_id not in self.data.get("sessions", {}):
            logger.warning("Нет активной сессии для завершения")
            return

        session_data = self.data["sessions"][self.session_id]
        session_data["status"] = status
        session_data["ended_at"] = datetime.now().isoformat()

        # Очищаем текущую сессию
        self.data["current_session"] = None
        self.session_id = None
        self.save()
        logger.info(f"Сессия индексации завершена со статусом: {status}")

    def update_file(self, file_path: Path, status: str, error_msg: str = None, skip_counters: bool = False):
        """Обновить статус файла (pending|loading|chunking|chunked|indexing|multilevel_indexing|indexed|error).

        Args:
            file_path (Path): Путь к файлу
            status (str): Новый статус
            error_msg (str, optional): Сообщение об ошибке
            skip_counters (bool): Пропустить обновление счетчиков (для уже проиндексированных файлов)
        """
        if not self.session_id:
            logger.warning("Нет активной сессии индексации")
            return

        str_path = str(file_path.resolve())
        session_data = self.data["sessions"][self.session_id]

        # Обновляем информацию о файле
        if str_path not in session_data["files"]:
            session_data["files"][str_path] = {
                "status": "pending",
                "updated_at": None,
                "error_message": None
            }

        file_data = session_data["files"][str_path]
        old_status = file_data["status"]
        file_data["status"] = status
        file_data["updated_at"] = datetime.now().isoformat()

        if error_msg:
            file_data["error_message"] = error_msg

        # Обновляем текущий файл и этап
        if status in ["loading", "chunking", "indexing", "multilevel_indexing"]:
            session_data["current_file"] = file_path.name
            session_data["current_stage"] = status
            logger.info(f"Файл {file_path.name} перешел в статус: {status}")
        elif status == "chunked":
            # Файл разбит на чанки, готов к индексации в Qdrant
            logger.info(f"Файл {file_path.name} разбит на чанки и готов к индексации")
        elif status in ["indexed", "error"]:
            session_data["current_file"] = None
            session_data["current_stage"] = None
            logger.info(f"Файл {file_path.name} завершил обработку со статусом: {status}")

        # Обновляем счетчики - только при реальном изменении статуса и если не пропущен счетчик
        if not skip_counters and old_status != status:
            logger.debug(f"Обновление счетчиков для файла {file_path.name}: {old_status} -> {status}")
            if old_status == "pending" and status != "pending":
                # Файл начинает обрабатываться
                session_data["processed_files"] += 1

            if status == "indexed":
                # Инкрементируем только если файл не был indexed до этого
                if old_status != "indexed":
                    session_data["indexed_files"] += 1
            elif status == "error":
                session_data["error_files"] += 1

        self.save()

    def set_total_files(self, total: int):
        """Установить общее количество файлов для сессии."""
        if not self.session_id:
            logger.warning("Нет активной сессии индексации")
            return

        self.data["sessions"][self.session_id]["total_files"] = total
        self.save()

    def get_unprocessed_files(self, directory: Path) -> List[Path]:
        """Возвращает список файлов из директории, не отмеченных как indexed в текущей сессии."""
        if not self.session_id:
            logger.warning("Нет активной сессии индексации")
            return []

        session_data = self.data["sessions"][self.session_id]
        all_files = []

        # Get all .txt and .md files from directory
        for ext in ['.txt', '.md']:
            all_files.extend(directory.rglob(f"*{ext}"))
            all_files.extend(directory.rglob(f"*{ext.upper()}"))

        unprocessed = []
        for file_path in all_files:
            if file_path.is_file():
                str_path = str(file_path.resolve())
                # Include file if it's not in session files or status is not 'indexed'
                if str_path not in session_data["files"] or session_data["files"][str_path]["status"] != "indexed":
                    unprocessed.append(file_path)

        return unprocessed

    def get_file_status(self, file_path: Path) -> str:
        """Получить статус конкретного файла в текущей сессии."""
        if not self.session_id:
            return "no_session"

        str_path = str(file_path.resolve())
        session_data = self.data["sessions"][self.session_id]

        if str_path in session_data["files"]:
            return session_data["files"][str_path]["status"]
        return "pending"

    def get_file_status_from_any_session(self, file_path: Path, collection_name: str = None) -> str:
        """Получить статус файла из любой сессии (последний известный статус) с учетом коллекции."""
        str_path = str(file_path.resolve())
        latest_status = "pending"
        latest_time = None

        # Проверяем все сессии на предмет этого файла
        for session_id, session_data in self.data.get("sessions", {}).items():
            # Если указана коллекция, проверяем только сессии с этой коллекцией
            if collection_name and session_data.get("collection_name") != collection_name:
                continue

            if str_path in session_data.get("files", {}):
                file_data = session_data["files"][str_path]
                file_status = file_data.get("status", "pending")
                updated_at = file_data.get("updated_at", "")

                # Если статус "indexed" или "error", запоминаем его как последний известный
                if file_status in ["indexed", "error"] and updated_at:
                    # Преобразуем время для сравнения
                    try:
                        if latest_time is None or updated_at > latest_time:
                            latest_status = file_status
                            latest_time = updated_at
                    except:
                        # Если не можем сравнить время, просто запоминаем статус
                        latest_status = file_status

        return latest_status

    def get_session_progress(self) -> Dict[str, Any]:
        """Получить прогресс текущей сессии индексации."""
        # Принудительно перезагружаем данные для актуальности
        self.load()

        if not self.session_id:
            return {
                "session_id": None,
                "status": "no_session",
                "total_files": 0,
                "processed_files": 0,
                "indexed_files": 0,
                "error_files": 0,
                "pending_files": 0,
                "current_file": None,
                "current_stage": None,
                "files": {}
            }

        session_data = self.data["sessions"][self.session_id]
        total = session_data.get("total_files", 0)
        processed = session_data.get("processed_files", 0)
        indexed = session_data.get("indexed_files", 0)
        errors = session_data.get("error_files", 0)
        pending = total - processed
        current_file = session_data.get("current_file")
        current_stage = session_data.get("current_stage")

        # Минимальное логирование для отладки
        logger.debug(f"Progress data - session_id: {self.session_id}, status: {session_data.get('status', 'unknown')}")
        logger.debug(f"Total: {total}, Processed: {processed}, Indexed: {indexed}, Errors: {errors}")
        logger.debug(f"Current file: {current_file}, Current stage: {current_stage}")

        return {
            "session_id": self.session_id,
            "status": session_data.get("status", "unknown"),
            "started_at": session_data.get("started_at"),
            "ended_at": session_data.get("ended_at"),
            "total_files": total,
            "processed_files": processed,
            "indexed_files": indexed,
            "error_files": errors,
            "pending_files": pending,
            "current_file": current_file,
            "current_stage": current_stage,
            "collection_name": session_data.get("collection_name"),
            "files": session_data.get("files", {})
        }

    def get_all_sessions(self) -> Dict[str, Any]:
        """Получить информацию обо всех сессиях."""
        return {
            "current_session": self.session_id,
            "sessions": self.data.get("sessions", {})
        }

    def stop_indexing(self):
        """Остановить процесс индексации."""
        indexing_stop_event.set()
        # Также сохраняем флаг в сессию
        if self.session_id and self.session_id in self.data.get("sessions", {}):
            self.data["sessions"][self.session_id]["stop_requested"] = True
            self.save()
        logger.info("Установлен флаг остановки индексации")

    def reset_stop_flag(self):
        """Сбросить флаг остановки."""
        indexing_stop_event.clear()
        # Также сбрасываем флаг в сессии
        if self.session_id and self.session_id in self.data.get("sessions", {}):
            self.data["sessions"][self.session_id]["stop_requested"] = False
            self.save()
        logger.info("Сброшен флаг остановки индексации")

    def is_indexing_stopped(self) -> bool:
        """Проверить, остановлена ли индексация."""
        # Перезагружаем данные из файла для актуальности
        self.load()

        # Проверяем и threading event, и флаг в сессии
        threading_stopped = indexing_stop_event.is_set()

        # Проверяем флаг в сессии
        session_stopped = False
        if self.session_id and self.session_id in self.data.get("sessions", {}):
            session_stopped = self.data["sessions"][self.session_id].get("stop_requested", False)

        return threading_stopped or session_stopped

    def get_stop_event(self):
        """Получить событие остановки."""
        return indexing_stop_event

    def cleanup_old_sessions(self, keep_sessions: int = 10):
        """Очистить старые сессии, оставив только указанное количество."""
        if "sessions" not in self.data:
            return

        sessions = self.data["sessions"]
        session_ids = sorted(sessions.keys(), reverse=True)

        # Удаляем старые сессии, но оставляем текущую если она есть
        sessions_to_keep = session_ids[:keep_sessions]
        if self.session_id and self.session_id not in sessions_to_keep:
            sessions_to_keep.append(self.session_id)

        for session_id in list(sessions.keys()):
            if session_id not in sessions_to_keep:
                del sessions[session_id]

        self.save()
        logger.info(f"Очистка старых сессий завершена. Оставлено сессий: {len(sessions)}")

    def reset_all(self):
        """Сбросить все данные отслеживания."""
        self.data = {}
        self.session_id = None
        indexing_stop_event.clear()
        self.save()
        logger.info("Все данные отслеживания индексации сброшены")

    def remove_session(self, session_id: str):
        """Удалить конкретную сессию из отслеживания."""
        if "sessions" not in self.data or session_id not in self.data["sessions"]:
            logger.warning(f"Сессия {session_id} не найдена")
            return

        del self.data["sessions"][session_id]

        # Если удаляем текущую сессию, сбрасываем её
        if self.session_id == session_id:
            self.session_id = None
            self.data["current_session"] = None

        self.save()
        logger.info(f"Сессия {session_id} удалена из отслеживания")