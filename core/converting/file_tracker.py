"""Модуль для отслеживания состояния файлов во время конвертации."""

import json
from pathlib import Path
from typing import Dict, Any, List
import logging
import threading

logger = logging.getLogger(__name__)

# Глобальное событие для остановки конвертации
conversion_stop_event = threading.Event()

class FileTracker:
    """Отслеживает состояние файлов для конвертации."""
    
    def __init__(self, db_path: Path = Path("file_tracking.json")):
        self.db_path = db_path
        self.data: Dict[str, Any] = {}
        self.load()

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

    def update_file(self, file_path: Path, status: str):
        """Обновить статус файла (pending|converted|error)."""
        str_path = str(file_path.resolve())
        self.data[str_path] = {"status": status, "updated_at": str(Path(file_path).stat().st_mtime)}
        self.save()

    def get_unprocessed_files(self, directory: Path) -> List[Path]:
        """Возвращает список файлов из директории, не отмеченных как converted."""
        all_files = []
        # Get all supported files from directory
        for ext in ['.pdf', '.docx', '.pptx', '.xlsx', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.html', '.htm', '.md']:
            all_files.extend(directory.rglob(f"*{ext}"))
            all_files.extend(directory.rglob(f"*{ext.upper()}"))
        
        unprocessed = []
        for file_path in all_files:
            if file_path.is_file():
                str_path = str(file_path.resolve())
                # Include file if it's not in tracking data or status is not 'converted'
                if str_path not in self.data or self.data[str_path]["status"] != "converted":
                    unprocessed.append(file_path)
        
        return unprocessed

    def get_file_status(self, file_path: Path) -> str:
        """Получить статус конкретного файла."""
        str_path = str(file_path.resolve())
        if str_path in self.data:
            return self.data[str_path]["status"]
        return "pending"

    def get_all_files_status(self) -> Dict[str, Dict[str, Any]]:
        """Получить статусы всех отслеживаемых файлов."""
        return self.data.copy()

    def reset_all(self):
        """Сбросить все данные отслеживания."""
        self.data = {}
        self.save()

    def remove_file(self, file_path: Path):
        """Удалить файл из отслеживания."""
        str_path = str(file_path.resolve())
        if str_path in self.data:
            del self.data[str_path]
            self.save()

    def stop_conversion(self):
        """Остановить процесс конвертации."""
        conversion_stop_event.set()

    def reset_stop_flag(self):
        """Сбросить флаг остановки."""
        conversion_stop_event.clear()

    def is_conversion_stopped(self) -> bool:
        """Проверить, остановлена ли конвертация."""
        return conversion_stop_event.is_set()

    def get_stop_event(self):
        """Получить событие остановки."""
        return conversion_stop_event