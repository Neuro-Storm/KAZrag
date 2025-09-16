"""Тесты для проверки различных стратегий чанкинга."""

import pytest

from core.indexing.paragraph_chunker import ParagraphTextSplitter


class TestParagraphChunker:
    """Тесты для ParagraphTextSplitter."""

    def test_split_text_by_paragraphs(self):
        """Тест разделения текста по абзацам."""
        text = """First paragraph of text.

Second paragraph of text.

Third paragraph of text.

Fourth paragraph of text.

Fifth paragraph of text."""
        
        # Тест с настройками по умолчанию (3 абзаца в чанке, 1 абзац перекрытия)
        splitter = ParagraphTextSplitter()
        chunks = splitter.split_text(text)
        
        # Ожидаем 2 чанка:
        # 1. Первый, второй, третий абзац
        # 2. Третий, четвертый, пятый абзац (с перекрытием)
        assert len(chunks) == 2
        assert "First paragraph of text.\n\nSecond paragraph of text.\n\nThird paragraph of text." in chunks[0]
        assert "Third paragraph of text.\n\nFourth paragraph of text.\n\nFifth paragraph of text." in chunks[1]

    def test_split_text_by_paragraphs_custom_settings(self):
        """Тест разделения текста по абзацам с пользовательскими настройками."""
        text = """First paragraph of text.

Second paragraph of text.

Third paragraph of text.

Fourth paragraph of text.

Fifth paragraph of text."""
        
        # Тест с 2 абзацами в чанке и без перекрытия
        splitter = ParagraphTextSplitter(paragraphs_per_chunk=2, paragraph_overlap=0)
        chunks = splitter.split_text(text)
        
        # Ожидаем 3 чанка:
        # 1. Первый, второй абзац
        # 2. Третий, четвертый абзац
        # 3. Пятый абзац (последний)
        assert len(chunks) == 3
        assert "First paragraph of text.\n\nSecond paragraph of text." in chunks[0]
        assert "Third paragraph of text.\n\nFourth paragraph of text." in chunks[1]
        assert "Fifth paragraph of text." in chunks[2]

    def test_split_text_by_paragraphs_with_overlap(self):
        """Тест разделения текста по абзацам с перекрытием."""
        text = """First paragraph of text.

Second paragraph of text.

Third paragraph of text.

Fourth paragraph of text.

Fifth paragraph of text."""
        
        # Тест с 2 абзацами в чанке и 1 абзацем перекрытия
        splitter = ParagraphTextSplitter(paragraphs_per_chunk=2, paragraph_overlap=1)
        chunks = splitter.split_text(text)
        
        # Ожидаем 4 чанка:
        # 1. Первый, второй абзац
        # 2. Второй, третий абзац (с перекрытием)
        # 3. Третий, четвертый абзац (с перекрытием)
        # 4. Четвертый, пятый абзац (с перекрытием)
        assert len(chunks) == 4
        assert "First paragraph of text.\n\nSecond paragraph of text." in chunks[0]
        assert "Second paragraph of text.\n\nThird paragraph of text." in chunks[1]
        assert "Third paragraph of text.\n\nFourth paragraph of text." in chunks[2]
        assert "Fourth paragraph of text.\n\nFifth paragraph of text." in chunks[3]

    def test_split_text_empty(self):
        """Тест разделения пустого текста."""
        text = ""
        splitter = ParagraphTextSplitter()
        chunks = splitter.split_text(text)
        assert chunks == []

    def test_split_text_single_paragraph(self):
        """Тест разделения текста с одним абзацем."""
        text = "Single paragraph of text."
        splitter = ParagraphTextSplitter()
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert "Single paragraph of text." in chunks[0]

    def test_split_text_with_large_overlap(self):
        """Тест разделения текста с большим перекрытием."""
        text = """First paragraph of text.

Second paragraph of text.

Third paragraph of text."""
        
        # Тест с 2 абзацами в чанке и 2 абзацами перекрытия (равно количеству абзацев в чанке)
        splitter = ParagraphTextSplitter(paragraphs_per_chunk=2, paragraph_overlap=2)
        chunks = splitter.split_text(text)
        
        # Ожидаем 2 чанка (из-за механизма защиты от бесконечного цикла):
        # 1. Первый, второй абзац
        # 2. Второй, третий абзац
        assert len(chunks) == 2
        assert "First paragraph of text.\n\nSecond paragraph of text." in chunks[0]
        assert "Second paragraph of text.\n\nThird paragraph of text." in chunks[1]

    def test_split_text_with_zero_paragraphs_per_chunk(self):
        """Тест разделения текста с нулевым количеством абзацев в чанке."""
        text = """First paragraph of text.

Second paragraph of text."""
        
        # Тест с 0 абзацев в чанке
        splitter = ParagraphTextSplitter(paragraphs_per_chunk=0)
        chunks = splitter.split_text(text)
        
        # Ожидаем 1 чанк с всем текстом
        assert len(chunks) == 1
        assert "First paragraph of text.\n\nSecond paragraph of text." in chunks[0]

    def test_split_text_with_negative_overlap(self):
        """Тест разделения текста с отрицательным перекрытием."""
        text = """First paragraph of text.

Second paragraph of text.

Third paragraph of text."""
        
        # Тест с отрицательным перекрытием
        splitter = ParagraphTextSplitter(paragraphs_per_chunk=2, paragraph_overlap=-1)
        chunks = splitter.split_text(text)
        
        # Ожидаем 2 чанка без перекрытия (отрицательное значение должно быть обработано)
        assert len(chunks) == 2
        assert "First paragraph of text.\n\nSecond paragraph of text." in chunks[0]
        assert "Third paragraph of text." in chunks[1]

    def test_split_text_with_excessive_overlap(self):
        """Тест разделения текста с чрезмерным перекрытием."""
        text = """First paragraph of text.

Second paragraph of text.

Third paragraph of text."""
        
        # Тест с перекрытием больше, чем количество абзацев в чанке
        splitter = ParagraphTextSplitter(paragraphs_per_chunk=2, paragraph_overlap=5)
        chunks = splitter.split_text(text)
        
        # Ожидаем 2 чанка (перекрытие должно быть ограничено количеством абзацев в чанке)
        assert len(chunks) == 2
        assert "First paragraph of text.\n\nSecond paragraph of text." in chunks[0]
        assert "Second paragraph of text.\n\nThird paragraph of text." in chunks[1]

    @pytest.mark.parametrize("paragraphs_per_chunk,paragraph_overlap,expected_chunks", [
        (1, 0, 5),  # Один абзац в чанке без перекрытия
        (5, 0, 1),  # Все абзацы в одном чанке
        (3, 2, 3),  # Три абзаца в чанке с двумя перекрытиями
        (2, 1, 4),  # Два абзаца в чанке с одним перекрытием
    ])
    def test_split_text_parametrized(self, paragraphs_per_chunk, paragraph_overlap, expected_chunks):
        """Параметризованный тест разделения текста по абзацам."""
        text = """First paragraph of text.

Second paragraph of text.

Third paragraph of text.

Fourth paragraph of text.

Fifth paragraph of text."""
        
        splitter = ParagraphTextSplitter(
            paragraphs_per_chunk=paragraphs_per_chunk,
            paragraph_overlap=paragraph_overlap
        )
        chunks = splitter.split_text(text)
        assert len(chunks) == expected_chunks