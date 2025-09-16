"""Тесты для многоуровневого чанкинга."""

import pytest
from langchain_core.documents import Document

from core.indexing.multilevel_chunker import (
    MultiLevelChunker,
    create_flexible_multilevel_chunker,
)


class TestMultiLevelChunker:
    """Тесты для MultiLevelChunker."""

    def setup_method(self):
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
        assert len(multilevel_chunks) > 0
        
        # Проверяем структуру первого чанка
        first_chunk = multilevel_chunks[0]
        assert "macro_chunk" in first_chunk
        assert "micro_chunks" in first_chunk
        assert "chunk_index" in first_chunk
        
        # Проверяем, что есть микро-чанки
        assert len(first_chunk["micro_chunks"]) > 0

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
        assert len(multilevel_chunks) > 0
        
        # Проверяем микро-чанки по предложениям
        first_chunk = multilevel_chunks[0]
        assert len(first_chunk["micro_chunks"]) > 0

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
        assert len(multilevel_chunks) > 0
        
        # Проверяем микро-чанки по символам
        first_chunk = multilevel_chunks[0]
        assert len(first_chunk["micro_chunks"]) > 0

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
        assert len(multilevel_chunks) > 0
        
        # Проверяем микро-чанки по абзацам
        first_chunk = multilevel_chunks[0]
        assert len(first_chunk["micro_chunks"]) > 0

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
        assert len(vectors) > 0
        
        # Первый вектор должен быть оригинальным текстом
        assert vectors[0] == self.sample_text

    def test_empty_document(self):
        """Тест обработки пустого документа."""
        empty_document = Document(page_content="", metadata={"source": "empty.txt"})
        chunker = MultiLevelChunker()
        
        multilevel_chunks = chunker.create_multilevel_chunks([empty_document])
        
        # Для пустого документа должно быть 0 чанков
        assert len(multilevel_chunks) == 0

    def test_short_document(self):
        """Тест обработки короткого документа."""
        short_text = "This is a short document."
        short_document = Document(page_content=short_text, metadata={"source": "short.txt"})
        chunker = MultiLevelChunker()
        
        multilevel_chunks = chunker.create_multilevel_chunks([short_document])
        
        # Должен быть создан хотя бы один чанк
        assert len(multilevel_chunks) > 0

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
        assert len(multilevel_chunks1) > 0
        
        # Тест с абзацами и предложениями
        chunker2 = create_flexible_multilevel_chunker(
            macro_strategy="paragraph",
            macro_size=3,
            micro_strategy="sentence",
            micro_size=2
        )
        
        multilevel_chunks2 = chunker2.create_multilevel_chunks([self.document])
        assert len(multilevel_chunks2) > 0

    def test_create_multilevel_chunks_with_empty_documents(self):
        """Тест создания многоуровневых чанков с пустым списком документов."""
        chunker = MultiLevelChunker()
        multilevel_chunks = chunker.create_multilevel_chunks([])
        assert len(multilevel_chunks) == 0

    def test_create_multilevel_chunks_with_invalid_strategy(self):
        """Тест создания многоуровневых чанков с недопустимой стратегией."""
        # Тест с недопустимой стратегией для макро-чанков
        chunker = MultiLevelChunker(
            macro_chunk_strategy="invalid_strategy",
            micro_chunk_strategy="character",
            micro_chunk_size=50
        )
        
        # Должен возникнуть ValueError
        with pytest.raises(ValueError):
            chunker.create_multilevel_chunks([self.document])

    def test_create_multilevel_chunks_with_zero_sizes(self):
        """Тест создания многоуровневых чанков с нулевыми размерами."""
        # Тест с нулевым размером макро-чанков
        chunker = MultiLevelChunker(
            macro_chunk_strategy="character",
            macro_chunk_size=0,
            micro_chunk_strategy="character",
            micro_chunk_size=50
        )
        
        multilevel_chunks = chunker.create_multilevel_chunks([self.document])
        # Должен быть создан хотя бы один чанк
        assert len(multilevel_chunks) > 0

    def test_create_multilevel_chunks_preserve_metadata(self):
        """Тест сохранения метаданных в многоуровневых чанках."""
        chunker = MultiLevelChunker(
            macro_chunk_strategy="paragraph",
            macro_paragraphs_per_chunk=2,
            micro_chunk_strategy="sentence",
            micro_sentences_per_chunk=3
        )
        
        multilevel_chunks = chunker.create_multilevel_chunks([self.document])
        
        # Проверяем, что метаданные сохранены
        first_chunk = multilevel_chunks[0]
        assert "metadata" in first_chunk
        assert first_chunk["metadata"]["source"] == "test.txt"
        assert first_chunk["metadata"]["author"] == "test_author"

    @pytest.mark.parametrize("macro_strategy,micro_strategy", [
        ("character", "character"),
        ("paragraph", "sentence"),
        ("sentence", "character"),
        ("paragraph", "paragraph"),
    ])
    def test_create_multilevel_chunks_parametrized_strategies(self, macro_strategy, micro_strategy):
        """Параметризованный тест создания многоуровневых чанков с разными стратегиями."""
        chunker = MultiLevelChunker(
            macro_chunk_strategy=macro_strategy,
            macro_chunk_size=200 if macro_strategy == "character" else None,
            macro_paragraphs_per_chunk=3 if macro_strategy == "paragraph" else None,
            macro_sentences_per_chunk=4 if macro_strategy == "sentence" else None,
            micro_chunk_strategy=micro_strategy,
            micro_chunk_size=50 if micro_strategy == "character" else None,
            micro_sentences_per_chunk=2 if micro_strategy == "sentence" else None,
            micro_paragraphs_per_chunk=2 if micro_strategy == "paragraph" else None
        )
        
        multilevel_chunks = chunker.create_multilevel_chunks([self.document])
        assert len(multilevel_chunks) > 0
        assert len(multilevel_chunks[0]["micro_chunks"]) > 0