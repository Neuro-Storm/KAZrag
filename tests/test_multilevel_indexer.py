"""Тесты для многоуровневого индексатора."""

import unittest
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from core.indexing.multilevel_indexer import MultiLevelIndexer
from config.settings import Config


class TestMultiLevelIndexer(unittest.TestCase):
    """Тесты для MultiLevelIndexer."""

    def setUp(self):
        """Настройка тестов."""
        # Создаем тестовую конфигурацию
        self.config = Config()
        self.config.collection_name = "test_collection"
        self.config.force_recreate = True
        self.config.index_dense = True
        self.config.index_bm25 = False
        self.config.index_hybrid = False
        
        # Создаем тестовый документ
        self.sample_text = """This is the first paragraph of our test document. 
It contains some sample text to demonstrate multilevel chunking.

This is the second paragraph of our test document. 
It also contains sample text for demonstration purposes.

This is the third paragraph of our test document. 
We are showing how the multilevel chunker works with multiple paragraphs."""

        self.document = Document(
            page_content=self.sample_text,
            metadata={"source": "test.txt", "author": "test_author"}
        )

    @patch('core.indexing.multilevel_indexer.get_qdrant_client')
    @patch('core.indexing.multilevel_indexer.EmbeddingManager')
    def test_index_documents_multilevel(self, mock_embedding_manager_class, mock_get_qdrant_client):
        """Тест индексации документов с многоуровневым чанкингом."""
        # Настройка моков
        mock_client = Mock()
        mock_get_qdrant_client.return_value = mock_client
        
        # Настройка мока для EmbeddingManager
        mock_embedding_manager_instance = Mock()
        mock_embedding_manager_class.get_instance.return_value = mock_embedding_manager_instance
        mock_embedding_manager_instance.embed_texts.return_value = [
            [0.1, 0.2, 0.3],  # Вектор для макро-чанка
            [0.4, 0.5, 0.6],  # Вектор для микро-чанка 1
            [0.7, 0.8, 0.9]   # Вектор для микро-чанка 2
        ]
        mock_embedding_manager_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Создаем индексатор
        indexer = MultiLevelIndexer(self.config)
        
        # Выполняем индексацию
        stats = indexer.index_documents_multilevel([self.document])
        
        # Проверяем результаты
        self.assertIn("macro_chunks", stats)
        self.assertIn("total_vectors", stats)
        self.assertIn("micro_vectors_per_chunk", stats)
        
        # Проверяем, что были вызваны необходимые методы
        mock_client.upload_points.assert_called_once()
        
        # Проверяем структуру переданных точек
        call_args = mock_client.upload_points.call_args
        self.assertIsNotNone(call_args)
        
        # Получаем аргументы вызова
        args, kwargs = call_args
        points = kwargs.get('points', []) if kwargs else (args[1] if len(args) > 1 else [])
        
        # Проверяем, что точки были созданы
        self.assertGreater(len(points), 0)
        
        # Проверяем структуру первой точки
        if points:
            point = points[0]
            self.assertIn("id", point)
            self.assertIn("vector", point)
            self.assertIn("payload", point)
            
            # Проверяем, что вектор является списком (мультивектор)
            self.assertIsInstance(point["vector"], list)
            
            # Проверяем наличие необходимых полей в payload
            payload = point["payload"]
            self.assertIn("content", payload)
            self.assertIn("metadata", payload)
            self.assertIn("chunk_index", payload)
            self.assertIn("total_micro_chunks", payload)
            self.assertIn("micro_contents", payload)

    @patch('core.indexing.multilevel_indexer.get_qdrant_client')
    @patch('core.indexing.multilevel_indexer.EmbeddingManager')
    @patch('core.indexing.multilevel_indexer.SparseEmbeddingAdapter')
    def test_index_documents_multilevel_hybrid(self, mock_sparse_adapter_class, mock_embedding_manager_class, mock_get_qdrant_client):
        """Тест гибридной индексации документов с многоуровневым чанкингом."""
        # Настройка конфигурации для гибридного режима
        self.config.index_bm25 = True
        self.config.index_hybrid = True
        
        # Настройка моков
        mock_client = Mock()
        mock_get_qdrant_client.return_value = mock_client
        
        # Настройка мока для EmbeddingManager
        mock_embedding_manager_instance = Mock()
        mock_embedding_manager_class.get_instance.return_value = mock_embedding_manager_instance
        mock_embedding_manager_instance.embed_texts.return_value = [
            [0.1, 0.2, 0.3],  # Вектор для макро-чанка
            [0.4, 0.5, 0.6],  # Вектор для микро-чанка 1
            [0.7, 0.8, 0.9]   # Вектор для микро-чанка 2
        ]
        mock_embedding_manager_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Настройка мока для SparseEmbeddingAdapter
        mock_sparse_adapter_instance = Mock()
        mock_sparse_adapter_class.return_value = mock_sparse_adapter_instance
        mock_sparse_vector = Mock()
        mock_sparse_vector.indices = [1, 5, 10]
        mock_sparse_vector.values = [0.5, 0.3, 0.8]
        mock_sparse_adapter_instance.embed_query.return_value = mock_sparse_vector
        
        # Создаем индексатор
        indexer = MultiLevelIndexer(self.config)
        
        # Выполняем индексацию
        stats = indexer.index_documents_multilevel([self.document])
        
        # Проверяем результаты
        self.assertIn("macro_chunks", stats)
        self.assertIn("total_vectors", stats)
        self.assertIn("micro_vectors_per_chunk", stats)
        
        # Проверяем, что были вызваны необходимые методы
        mock_client.upload_points.assert_called_once()
        
        # Проверяем структуру переданных точек
        call_args = mock_client.upload_points.call_args
        self.assertIsNotNone(call_args)
        
        # Получаем аргументы вызова
        args, kwargs = call_args
        points = kwargs.get('points', []) if kwargs else (args[1] if len(args) > 1 else [])
        
        # Проверяем, что точки были созданы
        self.assertGreater(len(points), 0)
        
        # Проверяем структуру первой точки в гибридном режиме
        if points:
            point = points[0]
            self.assertIn("id", point)
            self.assertIn("vector", point)
            self.assertIn("payload", point)
            
            # В гибридном режиме вектор должен быть словарем
            self.assertIsInstance(point["vector"], dict)
            self.assertIn("dense_vector", point["vector"])
            self.assertIn("sparse_vector", point["vector"])
            
            # Проверяем, что dense_vector является списком (мультивектор)
            self.assertIsInstance(point["vector"]["dense_vector"], list)
            
            # Проверяем структуру sparse_vector
            sparse_vector = point["vector"]["sparse_vector"]
            self.assertIsNotNone(sparse_vector)
            self.assertIn("indices", sparse_vector)
            self.assertIn("values", sparse_vector)
            
            # Проверяем наличие необходимых полей в payload
            payload = point["payload"]
            self.assertIn("content", payload)
            self.assertIn("metadata", payload)
            self.assertIn("chunk_index", payload)
            self.assertIn("total_micro_chunks", payload)
            self.assertIn("micro_contents", payload)


if __name__ == '__main__':
    unittest.main()