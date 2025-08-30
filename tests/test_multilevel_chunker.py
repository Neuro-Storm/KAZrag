"""Тесты для многоуровневого чанкинга."""

import unittest

from langchain_core.documents import Document

from core.indexing.multilevel_chunker import (
    MultiLevelChunker,
    create_flexible_multilevel_chunker,
)


class TestMultiLevelChunker(unittest.TestCase):
    """Тесты для MultiLevelChunker."""

    def setUp(self):
        """Настройка тестов."""
        self.sample_text = """This is the first paragraph of our test document. 
It contains some sample text to demonstrate multilevel chunking.

This is the second paragraph of our test document. 
It also contains sample text for demonstration purposes.

This is the third paragraph of our test document. 
We are showing how the multilevel chunker works with multiple paragraphs.

This is the fourth paragraph of our test document. 
It continues our demonstration of multilevel chunking.

This is the fifth paragraph of our test document. 
This is the final paragraph in our sample text."""

        self.document = Document(
            page_content=self.sample_text,
            metadata={"source": "test.txt", "author": "test_author"}
        )

    def test_create_multilevel_chunks_character_character(self):
        """Тест создания многоуровневых чанков с чанкингом по символам на обоих уровнях."""
        chunker = MultiLevelChunker(
            macro_chunk_strategy="character",
            macro_chunk_size=200,
            macro_chunk_overlap=50,
            micro_chunk_strategy="character",
            micro_chunk_size=50,
            micro_chunk_overlap=10
        )
        
        multilevel_chunks = chunker.create_multilevel_chunks([self.document])
        
        # Проверяем, что созданы макро-чанки
        self.assertGreater(len(multilevel_chunks), 0)
        
        # Проверяем структуру первого чанка
        first_chunk = multilevel_chunks[0]
        self.assertIn("macro_chunk", first_chunk)
        self.assertIn("micro_chunks", first_chunk)
        self.assertIn("chunk_index", first_chunk)
        
        # Проверяем, что есть микро-чанки
        self.assertGreater(len(first_chunk["micro_chunks"]), 0)

    def test_create_multilevel_chunks_paragraph_sentence(self):
        """Тест создания многоуровневых чанков с чанкингом по абзацам (макро) и предложениям (микро)."""
        chunker = MultiLevelChunker(
            macro_chunk_strategy="paragraph",
            macro_paragraphs_per_chunk=3,
            macro_paragraph_overlap=1,
            micro_chunk_strategy="sentence",
            micro_sentences_per_chunk=2,
            micro_sentence_overlap=1
        )
        
        multilevel_chunks = chunker.create_multilevel_chunks([self.document])
        
        # Проверяем, что созданы макро-чанки
        self.assertGreater(len(multilevel_chunks), 0)
        
        # Проверяем микро-чанки по предложениям
        first_chunk = multilevel_chunks[0]
        self.assertGreater(len(first_chunk["micro_chunks"]), 0)

    def test_create_multilevel_chunks_sentence_character(self):
        """Тест создания многоуровневых чанков с чанкингом по предложениям (макро) и символам (микро)."""
        chunker = MultiLevelChunker(
            macro_chunk_strategy="sentence",
            macro_sentences_per_chunk=4,
            macro_sentence_overlap=1,
            micro_chunk_strategy="character",
            micro_chunk_size=100,
            micro_chunk_overlap=10
        )
        
        multilevel_chunks = chunker.create_multilevel_chunks([self.document])
        
        # Проверяем, что созданы макро-чанки
        self.assertGreater(len(multilevel_chunks), 0)
        
        # Проверяем микро-чанки по символам
        first_chunk = multilevel_chunks[0]
        self.assertGreater(len(first_chunk["micro_chunks"]), 0)

    def test_create_multilevel_chunks_paragraph_paragraph(self):
        """Тест создания многоуровневых чанков с чанкингом по абзацам на обоих уровнях."""
        chunker = MultiLevelChunker(
            macro_chunk_strategy="paragraph",
            macro_paragraphs_per_chunk=4,
            macro_paragraph_overlap=1,
            micro_chunk_strategy="paragraph",
            micro_paragraphs_per_chunk=2,
            micro_paragraph_overlap=1
        )
        
        multilevel_chunks = chunker.create_multilevel_chunks([self.document])
        
        # Проверяем, что созданы макро-чанки
        self.assertGreater(len(multilevel_chunks), 0)
        
        # Проверяем микро-чанки по абзацам
        first_chunk = multilevel_chunks[0]
        self.assertGreater(len(first_chunk["micro_chunks"]), 0)

    def test_get_all_vectors_for_chunk(self):
        """Тест получения всех векторов для чанка."""
        chunker = MultiLevelChunker(
            macro_chunk_strategy="character",
            macro_chunk_size=200,
            macro_chunk_overlap=50,
            micro_chunk_strategy="character",
            micro_chunk_size=50,
            micro_chunk_overlap=10
        )
        
        # Получаем все векторы для тестового текста
        vectors = chunker.get_all_vectors_for_chunk(self.sample_text)
        
        # Должен быть как минимум один вектор (макро-чанк)
        self.assertGreater(len(vectors), 0)
        
        # Первый вектор должен быть оригинальным текстом
        self.assertEqual(vectors[0], self.sample_text)

    def test_empty_document(self):
        """Тест обработки пустого документа."""
        empty_document = Document(page_content="", metadata={"source": "empty.txt"})
        chunker = MultiLevelChunker()
        
        multilevel_chunks = chunker.create_multilevel_chunks([empty_document])
        
        # Для пустого документа должно быть 0 чанков
        self.assertEqual(len(multilevel_chunks), 0)

    def test_short_document(self):
        """Тест обработки короткого документа."""
        short_text = "This is a short document."
        short_document = Document(page_content=short_text, metadata={"source": "short.txt"})
        chunker = MultiLevelChunker()
        
        multilevel_chunks = chunker.create_multilevel_chunks([short_document])
        
        # Должен быть создан хотя бы один чанк
        self.assertGreater(len(multilevel_chunks), 0)

    def test_flexible_multilevel_chunker(self):
        """Тест гибкого создателя многоуровневого чанкера."""
        # Тест с символами на обоих уровнях
        chunker1 = create_flexible_multilevel_chunker(
            macro_strategy="character",
            macro_size=300,
            micro_strategy="character",
            micro_size=100
        )
        
        multilevel_chunks1 = chunker1.create_multilevel_chunks([self.document])
        self.assertGreater(len(multilevel_chunks1), 0)
        
        # Тест с абзацами и предложениями
        chunker2 = create_flexible_multilevel_chunker(
            macro_strategy="paragraph",
            macro_size=3,
            micro_strategy="sentence",
            micro_size=2
        )
        
        multilevel_chunks2 = chunker2.create_multilevel_chunks([self.document])
        self.assertGreater(len(multilevel_chunks2), 0)


if __name__ == '__main__':
    unittest.main()