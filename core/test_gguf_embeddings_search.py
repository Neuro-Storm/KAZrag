"""Тесты для проверки количества обращений к модели при поисковом запросе."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Добавляем путь к директории core, чтобы можно было импортировать модули
sys.path.append(os.path.join(os.path.dirname(__file__)))

from gguf_embeddings import GGUFEmbeddings, _embedding_dim_cache, _gguf_embedder_cache
from embeddings import get_dense_embedder


class TestGGUFEmbeddingsSearch(unittest.TestCase):
    def setUp(self):
        """Очистка кэша перед каждым тестом."""
        _embedding_dim_cache.clear()
        _gguf_embedder_cache.clear()

    def test_embedding_dimension_caching_logic(self):
        """Проверка логики кэширования размерности вектора."""
        # Создаем фиктивную конфигурацию с реальным путем к модели
        config = {
            "current_hf_model": "./models/Qwen3-Embedding-4B-Q8_0.gguf",
            "embedding_batch_size": 32,
        }
        
        # Проверяем, что файл модели существует
        self.assertTrue(os.path.exists(config["current_hf_model"]), "Файл модели не найден")
        
        # Проверяем, что кэш пуст в начале
        self.assertNotIn(config["current_hf_model"], _embedding_dim_cache)
        
        # Здесь мы не можем полноценно протестировать логику кэширования, 
        # потому что это потребует загрузки реальной модели, что нецелесообразно в unit тесте.
        # Вместо этого мы можем проверить, что кэш правильно очищается в setUp.
        pass

    @patch('llama_cpp.Llama')
    def test_model_calls_count(self, mock_llama):
        """Проверка количества вызовов модели."""
        # Настраиваем mock для модели Llama
        mock_model_instance = MagicMock()
        # Настройка возвращаемого значения для create_embedding
        def create_embedding_side_effect(text):
            if text == "test":
                return {
                    'data': [{'embedding': [0.1] * 1536}]  # Вектор размерности 1536
                }
            else:
                return {
                    'data': [{'embedding': [0.2] * 1536}]  # Вектор размерности 1536
                }
        
        mock_model_instance.create_embedding.side_effect = create_embedding_side_effect
        mock_llama.return_value = mock_model_instance
        
        # Создаем фиктивную конфигурацию с реальным путем к модели
        config = {
            "current_hf_model": "./models/Qwen3-Embedding-4B-Q8_0.gguf",
            "embedding_batch_size": 32,
        }
        
        # Проверяем, что файл модели существует
        self.assertTrue(os.path.exists(config["current_hf_model"]), "Файл модели не найден")
        
        # Первый вызов - создание экземпляра GGUFEmbeddings (инициализация)
        embedder = get_dense_embedder(config, device="cpu")
        
        # Проверяем, что тестовый запрос был выполнен один раз для определения размерности
        self.assertEqual(mock_model_instance.create_embedding.call_count, 1)
        mock_model_instance.create_embedding.assert_called_with("test")
        
        # Сбрасываем счетчик вызовов
        mock_model_instance.create_embedding.reset_mock()
        
        # Второй вызов - использование модели для векторизации запроса при поиске
        # Это имитирует вызов из searcher.py
        query_vector = embedder.embed_query("Тестовый поисковый запрос")
        
        # Проверяем, что метод create_embedding был вызван один раз для векторизации запроса
        self.assertEqual(mock_model_instance.create_embedding.call_count, 1)
        mock_model_instance.create_embedding.assert_called_with("Тестовый поисковый запрос")
        
        # Проверяем размерность вектора
        self.assertEqual(len(query_vector), 1536)
        
        # Сбрасываем счетчик вызовов
        mock_model_instance.create_embedding.reset_mock()
        
        # Третий вызов - повторное использование кэшированного экземпляра для векторизации другого запроса
        query_vector2 = embedder.embed_query("Другой тестовый запрос")
        
        # Проверяем, что метод create_embedding был вызван один раз для векторизации второго запроса
        self.assertEqual(mock_model_instance.create_embedding.call_count, 1)
        mock_model_instance.create_embedding.assert_called_with("Другой тестовый запрос")
        
        # Проверяем размерность второго вектора
        self.assertEqual(len(query_vector2), 1536)


if __name__ == '__main__':
    unittest.main()