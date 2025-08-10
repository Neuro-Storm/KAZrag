"""Тесты для проверки количества обращений к модели при поиске."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Добавляем путь к директории проекта, чтобы можно было импортировать модули
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main import app
from fastapi.testclient import TestClient


class TestModelCallsDuringSearch(unittest.TestCase):
    def setUp(self):
        """Создание тестового клиента перед каждым тестом."""
        self.client = TestClient(app)

    @patch('core.searcher.get_dense_embedder')
    @patch('web.search_app.get_qdrant_client')
    def test_model_calls_during_search(self, mock_get_qdrant_client, mock_get_dense_embedder):
        """Проверка количества обращений к модели при поиске."""
        # Настраиваем mock для Qdrant клиента
        mock_client = MagicMock()
        mock_get_qdrant_client.return_value = mock_client
        mock_client.get_collections.return_value.collections = [
            MagicMock(name="ESKD4000GPU")
        ]
        
        # Настраиваем mock для get_dense_embedder
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 1536  # Вектор размерности 1536
        mock_get_dense_embedder.return_value = mock_embedder
        
        # Выполняем несколько поисковых запросов
        queries = [
            "первый тестовый запрос",
            "второй тестовый запрос", 
            "третий тестовый запрос"
        ]
        
        for query in queries:
            response = self.client.post("/", data={
                "query": query,
                "collection": "ESKD4000GPU",
                "search_device": "cpu",
                "k": "5"
            })
            self.assertEqual(response.status_code, 200)
        
        # Проверяем, что get_dense_embedder был вызван столько раз, сколько было запросов
        # (в реальности он может кэшироваться, но для каждого запроса searcher.py вызывает get_dense_embedder)
        self.assertEqual(mock_get_dense_embedder.call_count, len(queries))
        
        # Проверяем, что embed_query был вызван столько раз, сколько было запросов
        self.assertEqual(mock_embedder.embed_query.call_count, len(queries))
        
        # Проверяем, что каждый запрос был передан в embed_query
        expected_calls = [unittest.mock.call(query) for query in queries]
        mock_embedder.embed_query.assert_has_calls(expected_calls)


if __name__ == '__main__':
    unittest.main()