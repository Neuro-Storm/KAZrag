"""Модуль для работы с GGUF эмбеддерами."""

import logging
import os
from typing import List

from langchain_core.embeddings import Embeddings

from config.config_manager import ConfigManager
from config.settings import Config

from pathlib import Path

logger = logging.getLogger(__name__)

def get_gguf_model(model_name: str = None):
    """Получение GGUF модели с автоматическим скачиванием в папку models."""
    config_manager = ConfigManager.get_instance()
    config = config_manager.get()
    
    if model_name is None:
        model_name = config.rag_model_path
    
    local_path = Path(config.local_models_path / "gguf" / model_name)
    
    if local_path.exists():
        logger.info(f"Используется локальная GGUF модель: {local_path}")
        from llama_cpp import Llama
        return Llama(
            model_path=str(local_path),
            n_ctx=config.rag_context_size,
            n_gpu_layers=config.rag_gpu_layers,
            n_threads=config.rag_threads,
            verbose=False
        )
    else:
        logger.error(f"Локальная GGUF модель не найдена: {local_path}")
        # Для GGUF моделей нет автоматического скачивания, так как они обычно большие
        # и требуют ручного скачивания
        raise FileNotFoundError(f"GGUF model not found at {local_path}. Please download it manually.")

# Импортируем llama_cpp для правильной работы тестов
try:
    import importlib.util
    LLAMA_CPP_AVAILABLE = importlib.util.find_spec("llama_cpp") is not None
except ImportError:
    LLAMA_CPP_AVAILABLE = False


class GGUFEmbeddings(Embeddings):
    """Класс для работы с GGUF моделями через llama.cpp Python биндинги."""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Инициализация GGUF эмбеддера.
        
        Args:
            model_path: Путь к GGUF модели
            device: Устройство для вычислений ("cpu" или "cuda")
        """
        try:
            from llama_cpp import Llama
            
            # Загружаем конфигурацию для получения параметров
            config_manager = ConfigManager.get_instance()
            config = config_manager.get()
            
            logger.info(f"Инициализация GGUF эмбеддингов с моделью: {model_path}")
            logger.info(f"Контекст модели: {config.gguf_model_n_ctx}, количество потоков: {config.rag_threads}")
            
            # Проверяем, существует ли файл модели
            if not os.path.exists(model_path):
                # Если файл не найден, пытаемся найти его в стандартных местах
                # MODELS_DIR может быть установлена пользователем (кроссплатформенно)
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
            
            # Проверяем кэш размерности вектора
            global _embedding_dim_cache
            if model_path in _embedding_dim_cache:
                self.expected_dim = _embedding_dim_cache[model_path]
                logger.info(f"Размерность вектора загружена из кэша: {self.expected_dim}")
                # Инициализируем модель без тестового запроса
                n_gpu_layers = -1 if device == "cuda" else 0
                try:
                    self.model = Llama(
                        model_path=model_path,
                        n_gpu_layers=n_gpu_layers,  # Количество слоев на GPU (-1 для всех слоев)
                        n_ctx=config.gguf_model_n_ctx,  # Размер контекста из конфигурации
                        embedding=True,  # Включаем режим эмбеддингов
                        use_mmap=True,  # Используем mmap для экономии памяти
                        use_mlock=False,  # Не используем mlock, чтобы позволить свопинг
                        n_threads=config.rag_threads,  # Используем количество потоков из конфига
                        verbose=False  # Отключаем подробный вывод
                    )
                    logger.info(f"GGUF модель инициализирована для эмбеддингов, контекст: {config.gguf_model_n_ctx}")
                except Exception as e:
                    # Если размер контекста слишком велик, выводим информативное сообщение
                    if "n_ctx_per_seq" in str(e) or "context overflow" in str(e).lower():
                        error_msg = (f"Контекст GGUF модели ({config.gguf_model_n_ctx}) больше, "
                                    f"чем контекст, на котором модель была обучена. Это приводит к проблемам с индексацией. "
                                    f"Пожалуйста, измените параметр 'embedding.gguf.model_n_ctx' в настройках на 512 или "
                                    f"другое значение, соответствующее вашей модели. Ошибка: {str(e)}")
                        logger.error(error_msg)
                        raise ValueError(error_msg) from e
                    else:
                        logger.error(f"Ошибка при инициализации GGUF модели: {e}")
                        raise
            else:
                # Инициализируем модель для определения размерности вектора
                n_gpu_layers = -1 if device == "cuda" else 0
                try:
                    self.model = Llama(
                        model_path=model_path,
                        n_gpu_layers=n_gpu_layers,  # Количество слоев на GPU (-1 для всех слоев)
                        n_ctx=config.gguf_model_n_ctx,  # Размер контекста из конфигурации
                        embedding=True,  # Включаем режим эмбеддингов
                        use_mmap=True,  # Используем mmap для экономии памяти
                        use_mlock=False,  # Не используем mlock, чтобы позволить свопинг
                        n_threads=config.rag_threads,  # Используем количество потоков из конфига
                        verbose=False  # Отключаем подробный вывод
                    )
                    logger.info(f"GGUF модель инициализирована для эмбеддингов, контекст: {config.gguf_model_n_ctx}")
                except Exception as e:
                    # Если размер контекста слишком велик, выводим информативное сообщение
                    if "n_ctx_per_seq" in str(e) or "context overflow" in str(e).lower():
                        error_msg = (f"Контекст GGUF модели ({config.gguf_model_n_ctx}) больше, "
                                    f"чем контекст, на котором модель была обучена. Это приводит к проблемам с индексацией. "
                                    f"Пожалуйста, измените параметр 'embedding.gguf.model_n_ctx' в настройках на 512 или "
                                    f"другое значение, соответствующее вашей модели. Ошибка: {str(e)}")
                        logger.error(error_msg)
                        raise ValueError(error_msg) from e
                    else:
                        logger.error(f"Ошибка при инициализации GGUF модели: {e}")
                        raise
                
                # Определяем ожидаемую размерность вектора на основе тестового запроса
                logger.info("Определение размерности вектора на основе тестового запроса")
                test_embedding = self.model.create_embedding("test")
                self.expected_dim = len(test_embedding['data'][0]['embedding'])
                logger.info(f"Размерность вектора определена: {self.expected_dim}")
                # Сохраняем размерность в кэш
                _embedding_dim_cache[model_path] = self.expected_dim
                logger.info(f"Размерность вектора сохранена в кэш для модели {model_path}")
            
            self.device = device
        except ImportError:
            logger.exception("Не удалось импортировать llama_cpp. Установите его с помощью `pip install llama-cpp-python`.")
            raise
        except Exception as e:
            logger.exception(f"Ошибка при загрузке GGUF модели: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Генерация эмбеддингов для списка документов."""
        logger.info(f"Начало генерации эмбеддингов для {len(texts)} документов")
        embeddings = []
        for i, text in enumerate(texts):
            logger.debug(f"Обработка документа {i+1}/{len(texts)}, длина: {len(text)} символов")
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        logger.info(f"Генерация эмбеддингов завершена для {len(texts)} документов")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Генерация эмбеддинга для одного запроса."""
        try:
            logger.info(f"Начало генерации эмбеддинга для текста длиной {len(text)} символов")
            
            # Получаем эмбеддинг через llama.cpp
            embedding_result = self.model.create_embedding(text)
            logger.info(f"Результат эмбеддинга получен, тип: {type(embedding_result)}")
            
            # Извлекаем вектор из результата - llama-cpp-python может возвращать эмбеддинги по-разному
            if isinstance(embedding_result, dict):
                # Стандартный формат: {'data': [{'embedding': [...]}, ...], 'model', 'object', ...}
                embedding_data = embedding_result.get('data', [])
                logger.info(f"Структура результата: {list(embedding_result.keys()) if isinstance(embedding_result, dict) else 'N/A'}")
                logger.info(f"Данные эмбеддинга: {len(embedding_data)} элементов")
                if len(embedding_data) > 0:
                    vector = embedding_data[0].get('embedding', [])
                else:
                    raise ValueError("No embedding data found in response")
            else:
                # Альтернативный формат: объект с атрибутом embedding
                logger.info(f"Результат не является словарем, атрибуты: {dir(embedding_result) if hasattr(embedding_result, '__dict__') else 'N/A'}")
                vector = getattr(embedding_result, 'embedding', []) if hasattr(embedding_result, 'embedding') else []
            
            if not vector:
                raise ValueError("Invalid embedding format - no embedding vector found")
            
            logger.info(f"Длина вектора эмбеддинга: {len(vector)}, ожидаемая длина: {self.expected_dim}")
            
            # Проверяем размерность вектора
            if len(vector) != self.expected_dim:
                logger.warning(f"Embedding dimension mismatch: {len(vector)} vs {self.expected_dim}, actual model dim: {len(vector)}")
                # Обновляем размерность, если она изменилась
                self.expected_dim = len(vector)
            
            return vector
        except Exception as e:
            logger.error(f"Ошибка при получении эмбеддинга для текста '{text[:50]}...': {e}")
            raise Exception(f"Ошибка при получении эмбеддинга: {e}")


# Кэш для GGUF embedder'а
_gguf_embedder_cache = {}
# Кэш для размерности вектора
_embedding_dim_cache = {}


def get_gguf_embedder(config: Config, device=None):
    """Получает или создает кэшированный экземпляр GGUFEmbeddings."""
    global _gguf_embedder_cache, _embedding_dim_cache
    model_name = config.current_hf_model
    batch_size = config.embedding_batch_size
    if device is None:
        device = "cpu"  # По умолчанию используем CPU для GGUF
    
    # Ключ кэша - кортеж (модель, устройство)
    cache_key = (model_name, device)
    
    # Проверяем, есть ли в кэше модель с такими параметрами
    if cache_key in _gguf_embedder_cache:
        # Проверяем, соответствует ли batch_size
        cached_embedder = _gguf_embedder_cache[cache_key]
        if getattr(cached_embedder, "_batch_size", None) == batch_size:
            return cached_embedder
    
    # Если модель не найдена в кэше или batch_size изменился, создаем новую
    embedder = GGUFEmbeddings(
        model_path=model_name,
        device=device
    )
    embedder._batch_size = batch_size  # Для отслеживания
    
    # Сохраняем модель в кэш
    _gguf_embedder_cache[cache_key] = embedder
    
    return embedder