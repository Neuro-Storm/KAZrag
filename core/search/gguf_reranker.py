"""Модуль для работы с локальными моделями reranker в формате GGUF через Sentence Transformers."""

import logging
import os
from typing import List, Tuple, Any
from sentence_transformers import CrossEncoder

from config.settings import Config

logger = logging.getLogger(__name__)

# Кэш для локальных reranker'ов
_reranker_cache = {}


class LocalReranker:
    """Класс для работы с локальной моделью reranker."""
    
    def __init__(self, model_path: str):
        """
        Инициализация локального reranker.
        
        Args:
            model_path: Путь к локальной модели reranker
        """
        try:
            # Проверяем, существует ли файл модели
            if not os.path.exists(model_path):
                # Если файл не найден, пытаемся найти его в стандартных местах
                # MODELS_DIR может быть установлена пользователем (кроссплатформенно)
                from pathlib import Path
                models_dir = os.getenv("MODELS_DIR", "")
                possible_paths = [
                    model_path,
                    Path("models") / model_path,
                    Path("data") / model_path,
                    Path("..") / "models" / model_path,
                ]
                
                # Добавляем путь из переменной окружения MODELS_DIR, если она установлена
                # Path() автоматически нормализует путь для текущей ОС
                if models_dir:
                    possible_paths.append(Path(models_dir) / model_path)
                
                found = False
                for path in possible_paths:
                    path_obj = Path(path) if not isinstance(path, Path) else path
                    if path_obj.exists():
                        model_path = str(path_obj)
                        found = True
                        break
                
                if not found:
                    raise FileNotFoundError(f"Файл модели {model_path} не найден. Попробуйте указать полный путь к файлу.")
            
            # Инициализируем модель
            self.model = CrossEncoder(model_path)
            logger.info(f"Локальная модель reranker загружена: {model_path}")
        except Exception as e:
            logger.exception(f"Ошибка при загрузке локальной модели reranker: {e}")
            raise
    
    def predict(self, pairs: List[List[str]]) -> List[float]:
        """Получает оценки релевантности для пар запрос-документ."""
        try:
            return self.model.predict(pairs).tolist()
        except Exception as e:
            logger.exception(f"Ошибка при предсказании reranker: {e}")
            raise


def get_local_reranker(config: Config) -> LocalReranker:
    """Получает или создает кэшированный экземпляр локального reranker."""
    global _reranker_cache
    model_name = config.reranker_model
    
    # Проверяем, есть ли в кэше модель с таким именем
    if model_name in _reranker_cache:
        return _reranker_cache[model_name]
    
    # Если модель не найдена в кэше, создаем новую
    reranker = LocalReranker(model_name)
    
    # Сохраняем модель в кэш
    _reranker_cache[model_name] = reranker
    
    return reranker