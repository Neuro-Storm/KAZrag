import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download, list_repo_files
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """Утилита для управления моделями."""
    
    @staticmethod
    def download_model(model_name: str, model_type: str, token: str = None):
        """Скачивание модели в соответствующую папку."""
        model_path = Path(f"./models/{model_type}/{model_name}")
        
        if model_path.exists():
            logger.info(f"Модель {model_name} уже существует в {model_path}")
            return model_path
        
        try:
            logger.info(f"Скачивание модели {model_name} в {model_path}...")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            downloaded_path = snapshot_download(
                repo_id=model_name,
                cache_dir="./models/huggingface_cache",
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
                token=token
            )
            
            logger.info(f"Модель успешно скачана в: {downloaded_path}")
            return downloaded_path
        except Exception as e:
            logger.error(f"Ошибка при скачивании модели {model_name}: {e}")
            raise
    
    @staticmethod
    def list_models(model_type: str):
        """Список моделей указанного типа."""
        model_dir = Path(f"./models/{model_type}")
        if not model_dir.exists():
            return []
        
        if model_type == "easyocr":
            # Для EasyOCR возвращаем файлы моделей (*.pth)
            return [f.name for f in model_dir.glob("*.pth")]
        else:
            # Для остальных типов возвращаем директории
            return [d.name for d in model_dir.iterdir() if d.is_dir()]
    
    @staticmethod
    def remove_model(model_name: str, model_type: str):
        """Удаление модели."""
        model_path = Path(f"./models/{model_type}/{model_name}")
        if model_path.exists():
            shutil.rmtree(model_path)
            logger.info(f"Модель {model_name} удалена")
            return True
        return False
    
    @staticmethod 
    @lru_cache(maxsize=128)
    def get_model_size(model_name: str, model_type: str):
        """Получение размера модели с кэшированием."""
        model_path = Path(f"./models/{model_type}/{model_name}")
        if not model_path.exists():
            return 0
        
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(model_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        
        return total_size / (1024 * 1024)  # Размер в МБ