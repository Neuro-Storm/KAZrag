"""Модуль для работы с GGUF эмбеддерами."""

import numpy as np
from typing import List
from langchain_core.embeddings import Embeddings
from config.settings import load_config
import os

# Импортируем llama_cpp для правильной работы тестов
try:
    import llama_cpp
    LLAMA_CPP_AVAILABLE = True
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
            
            # Проверяем, существует ли файл модели
            if not os.path.exists(model_path):
                # Если файл не найден, пытаемся найти его в стандартных местах
                possible_paths = [
                    model_path,
                    os.path.join("models", model_path),
                    os.path.join("data", model_path),
                    os.path.join("..", "models", model_path),
                    os.path.join("D:", "models", model_path),
                ]
                
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
                    n_ctx=4096,  # Размер контекста
                    embedding=True,  # Включаем режим эмбеддингов
                    verbose=False  # Отключаем подробный вывод
                )
            else:
                # Инициализируем модель для определения размерности вектора
                n_gpu_layers = -1 if device == "cuda" else 0
                self.model = Llama(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,  # Количество слоев на GPU (-1 для всех слоев)
                    n_ctx=4096,  # Размер контекста
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
            raise ImportError(
                "Не удалось импортировать llama_cpp. "
                "Установите его с помощью `pip install llama-cpp-python`."
            )
        except Exception as e:
            raise Exception(f"Ошибка при загрузке GGUF модели: {e}")
    
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
            if isinstance(embedding, dict) and "data" in embedding:
                # Обрабатываем вложенный список
                raw_vector = embedding["data"][0]["embedding"]
                # Если это вложенный список, извлекаем первый элемент
                if isinstance(raw_vector, list) and len(raw_vector) > 0:
                    if isinstance(raw_vector[0], list):
                        vector = raw_vector[0]
                    else:
                        vector = raw_vector
                else:
                    vector = raw_vector
            elif hasattr(embedding, 'embedding'):
                vector = embedding.embedding
            else:
                raise Exception("Не удалось извлечь эмбеддинг из результата модели")
            
            # Проверяем размерность вектора
            if len(vector) != self.expected_dim:
                print(f"Предупреждение: Размерность вектора {len(vector)} не соответствует ожидаемой {self.expected_dim}")
                # Если вектор больше, обрезаем его
                if len(vector) > self.expected_dim:
                    vector = vector[:self.expected_dim]
                # Если вектор меньше, дополняем нулями
                elif len(vector) < self.expected_dim:
                    vector = vector + [0.0] * (self.expected_dim - len(vector))
            
            return vector
        except Exception as e:
            raise Exception(f"Ошибка при получении эмбеддинга: {e}")


# Кэш для GGUF embedder'а
_gguf_embedder_cache = {}
# Кэш для размерности вектора
_embedding_dim_cache = {}


def get_gguf_embedder(config, device=None):
    """Получает или создает кэшированный экземпляр GGUFEmbeddings."""
    global _gguf_embedder_cache, _embedding_dim_cache
    model_name = config["current_hf_model"]
    batch_size = config.get("embedding_batch_size", 32)
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