"""Тесты для проверки кэширования размерности вектора в GGUFEmbeddings."""

import os
import sys
import unittest
import time

# Добавляем путь к директории core, чтобы можно было импортировать gguf_embeddings
sys.path.append(os.path.join(os.path.dirname(__file__)))

from gguf_embeddings import GGUFEmbeddings, _embedding_dim_cache, _gguf_embedder_cache


class TestGGUFEmbeddings(unittest.TestCase):
    def setUp(self):
        """Очистка кэша перед каждым тестом."""
        _embedding_dim_cache.clear()
        _gguf_embedder_cache.clear()

    def test_embedding_dimension_caching(self):
        """Проверка кэширования размерности вектора."""
        model_path = "./models/Qwen3-Embedding-4B-Q8_0.gguf"
        
        # Проверяем, что файл модели существует
        self.assertTrue(os.path.exists(model_path), "Файл модели не найден")
        
        # Первое создание экземпляра - должно выполнить тестовый запрос
        start_time = time.time()
        embedder1 = GGUFEmbeddings(model_path=model_path, device="cpu")
        first_creation_time = time.time() - start_time
        
        # Проверяем, что размерность сохранена в кэше
        self.assertIn(model_path, _embedding_dim_cache)
        expected_dim = _embedding_dim_cache[model_path]
        
        # Проверяем, что размерность имеет разумное значение
        self.assertGreater(expected_dim, 0)
        print(f"Размерность вектора: {expected_dim}")
        
        # Второе создание экземпляра - должно использовать кэш
        start_time = time.time()
        embedder2 = GGUFEmbeddings(model_path=model_path, device="cpu")
        second_creation_time = time.time() - start_time
        
        # Проверяем, что размерность все еще в кэше и не изменилась
        self.assertIn(model_path, _embedding_dim_cache)
        self.assertEqual(_embedding_dim_cache[model_path], expected_dim)
        
        # Проверяем, что размерность установлена в обоих экземплярах
        self.assertEqual(embedder1.expected_dim, expected_dim)
        self.assertEqual(embedder2.expected_dim, expected_dim)
        
        # Время второго создания должно быть значительно меньше первого
        # (это не строгая проверка, а скорее индикатор)
        print(f"Время первого создания: {first_creation_time:.4f} сек")
        print(f"Время второго создания: {second_creation_time:.4f} сек")
        
        # Выводим разницу во времени
        time_diff = first_creation_time - second_creation_time
        print(f"Разница во времени: {time_diff:.4f} сек")
        
        # Простая проверка, что второе создание быстрее (может не всегда выполняться в CI)
        # self.assertGreater(time_diff, 0, "Второе создание должно быть быстрее первого")


if __name__ == '__main__':
    unittest.main()