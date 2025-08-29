"""Тесты для проверки различных стратегий чанкинга."""

import unittest
from core.indexing.paragraph_chunker import ParagraphTextSplitter


class TestParagraphChunker(unittest.TestCase):
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
        self.assertEqual(len(chunks), 2)
        self.assertIn("First paragraph of text.\n\nSecond paragraph of text.\n\nThird paragraph of text.", chunks[0])
        self.assertIn("Third paragraph of text.\n\nFourth paragraph of text.\n\nFifth paragraph of text.", chunks[1])

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
        self.assertEqual(len(chunks), 3)
        self.assertIn("First paragraph of text.\n\nSecond paragraph of text.", chunks[0])
        self.assertIn("Third paragraph of text.\n\nFourth paragraph of text.", chunks[1])
        self.assertIn("Fifth paragraph of text.", chunks[2])

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
        self.assertEqual(len(chunks), 4)
        self.assertIn("First paragraph of text.\n\nSecond paragraph of text.", chunks[0])
        self.assertIn("Second paragraph of text.\n\nThird paragraph of text.", chunks[1])
        self.assertIn("Third paragraph of text.\n\nFourth paragraph of text.", chunks[2])
        self.assertIn("Fourth paragraph of text.\n\nFifth paragraph of text.", chunks[3])

    def test_split_text_empty(self):
        """Тест разделения пустого текста."""
        text = ""
        splitter = ParagraphTextSplitter()
        chunks = splitter.split_text(text)
        self.assertEqual(chunks, [])

    def test_split_text_single_paragraph(self):
        """Тест разделения текста с одним абзацем."""
        text = "Single paragraph of text."
        splitter = ParagraphTextSplitter()
        chunks = splitter.split_text(text)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Single paragraph of text.", chunks[0])

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
        self.assertEqual(len(chunks), 2)
        self.assertIn("First paragraph of text.\n\nSecond paragraph of text.", chunks[0])
        self.assertIn("Second paragraph of text.\n\nThird paragraph of text.", chunks[1])


if __name__ == '__main__':
    unittest.main()