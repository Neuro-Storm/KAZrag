"""Модуль для работы с GGUF эмбеддерами."""

import logging
from typing import List
from langchain_core.embeddings import Embeddings
from config.settings import load_config, Config
import os

logger = logging.getLogger(__name__)

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
            config = load_config()
            
            # Проверяем, существует ли файл модели
            if not os.path.exists(model_path):
                # Если файл не найден, пытаемся найти его в стандартных местах
                models_dir = os.getenv("MODELS_DIR", "")
                possible_paths = [
                    model_path,
                    os.path.join("models", model_path),
                    os.path.join("data", model_path),
                    os.path.join("..", "models", model_path),
                ]
                
                # Добавляем путь из переменной окружения MODELS_DIR, если она установлена
                if models_dir:
                    possible_paths.append(os.path.join(models_dir, model_path))
                
                found = False
                for path in possible_paths:
                    if os.path.exists(path):
                        model_path = path
                        found = True
                        break
                
                if not found:
                    raise FileNotFoundError(f"Файл модели {model_path} не найден. Попробуйте указать полный путь к файлу.")
            
            # Проверяем кэш размерности вектора
            global _embedding_dim_cache
            if model_path in _embedding_dim_cache:
                self.expected_dim = _embedding_dim_cache[model_path]
                # Инициализируем модель без тестового запроса
                n_gpu_layers = -1 if device == "cuda" else 0
                self.model = Llama(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,  # Количество слоев на GPU (-1 для всех слоев)
                    n_ctx=config.gguf_model_n_ctx,  # Размер контекста из конфигурации
                    embedding=True,  # Включаем режим эмбеддингов
                    verbose=False  # Отключаем подробный вывод
                )
            else:
                # Инициализируем модель для определения размерности вектора
                n_gpu_layers = -1 if device == "cuda" else 0
                self.model = Llama(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,  # Количество слоев на GPU (-1 для всех слоев)
                    n_ctx=config.gguf_model_n_ctx,  # Размер контекста из конфигурации
                    embedding=True,  # Включаем режим эмбеддингов
                    verbose=False  # Отключаем подробный вывод
                )
                # Определяем ожидаемую размерность вектора на основе тестового запроса
                test_embedding = self.model.create_embedding("test")
                self.expected_dim = len(test_embedding['data'][0]['embedding'])
                # Сохраняем размерность в кэш
                _embedding_dim_cache[model_path] = self.expected_dim
            
            self.device = device
        except ImportError:
            logger.exception("Не удалось импортировать llama_cpp. Установите его с помощью `pip install llama-cpp-python`.")
            raise
        except Exception as e:
            logger.exception(f"Ошибка при загрузке GGUF модели: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Генерация эмбеддингов для списка документов."""
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Генерация эмбеддинга для одного запроса."""
        try:
            # Получаем эмбеддинг через llama.cpp
            embedding = self.model.create_embedding(text)
            # Извлекаем вектор из результата
            vector = embedding.get('data', [{}])[0].get('embedding', []) or getattr(embedding, 'embedding', [])
            if not vector:
                raise ValueError("Invalid embedding format")
            
            # Проверяем размерность вектора
            if len(vector) != self.expected_dim:
                # logger.warning(f"Embedding dimension mismatch: {len(vector)} vs {self.expected_dim}")
                raise ValueError(f"Embedding dimension mismatch: {len(vector)} vs {self.expected_dim}")
            
            return vector
        except Exception as e:
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