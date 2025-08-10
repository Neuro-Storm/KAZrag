"""Тесты для проверки поисковых запросов к веб-приложению."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Добавляем путь к директории проекта, чтобы можно было импортировать модули
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main import app
from fastapi.testclient import TestClient


class TestSearchApp(unittest.TestCase):
    def setUp(self):
        """Создание тестового клиента перед каждым тестом."""
        self.client = TestClient(app)

    def test_get_search_page(self):
        """Проверка доступности главной страницы поиска."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        # Проверяем, что в ответе есть ключевые элементы страницы
        self.assertIn("Поиск по документам", response.text)  # Заголовок страницы
        self.assertIn("Найти", response.text)  # Кнопка поиска

    @patch('web.search_app.search_in_collection')
    @patch('web.search_app.get_qdrant_client')
    def test_search_request(self, mock_get_qdrant_client, mock_search_in_collection):
        """Проверка обработки поискового запроса."""
        # Настраиваем mock для Qdrant клиента
        mock_client = MagicMock()
        mock_get_qdrant_client.return_value = mock_client
        mock_client.get_collections.return_value.collections = [
            MagicMock(name="ESKD4000GPU")
        ]
        
        # Настраиваем mock для функции поиска
        mock_search_in_collection.return_value = [
            (MagicMock(page_content="Тестовый результат 1"), 0.9),
            (MagicMock(page_content="Тестовый результат 2"), 0.8)
        ]
        
        # Выполняем POST запрос с параметрами поиска
        response = self.client.post("/", data={
            "query": "тестовый запрос",
            "collection": "ESKD4000GPU",
            "search_device": "cpu",
            "k": "5"
        })
        
        # Проверяем, что запрос выполнен успешно
        self.assertEqual(response.status_code, 200)
        
        # Проверяем, что функция поиска была вызвана с правильными параметрами
        mock_search_in_collection.assert_called_once_with(
            "тестовый запрос", 
            "ESKD4000GPU", 
            "cpu", 
            5
        )
        
        # Проверяем, что в ответе есть результаты поиска
        self.assertIn("тестовый запрос", response.text)
        self.assertIn("Тестовый результат 1", response.text)
        self.assertIn("Тестовый результат 2", response.text)


if __name__ == '__main__':
    unittest.main()