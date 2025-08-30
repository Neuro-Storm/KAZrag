"""Тесты для проверки чанкинга по предложениям."""

import unittest

from core.indexing.sentence_chunker import SentenceTextSplitter


class TestSentenceChunker(unittest.TestCase):
    """Тесты для SentenceTextSplitter."""

    def test_split_text_by_sentences(self):
        """Тест разделения текста по предложениям."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence. Fifth sentence."
        
        # Тест с настройками по умолчанию (5 предложений в чанке, 1 предложение перекрытия)
        splitter = SentenceTextSplitter()
        chunks = splitter.split_text(text)
        
        # Ожидаем 1 чанк, так как у нас всего 5 предложений
        self.assertEqual(len(chunks), 1)
        # Проверяем, что чанк содержит все предложения
        self.assertIn("First sentence.", chunks[0])
        self.assertIn("Second sentence!", chunks[0])
        self.assertIn("Third sentence?", chunks[0])
        self.assertIn("Fourth sentence.", chunks[0])
        self.assertIn("Fifth sentence.", chunks[0])

    def test_split_text_by_sentences_custom_settings(self):
        """Тест разделения текста по предложениям с пользовательскими настройками."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence. Fifth sentence. Sixth sentence."
        
        # Тест с 3 предложениями в чанке и без перекрытия
        splitter = SentenceTextSplitter(sentences_per_chunk=3, sentence_overlap=0)
        chunks = splitter.split_text(text)
        
        # Ожидаем 2 чанка:
        # 1. Первые 3 предложения
        # 2. Последние 3 предложения
        self.assertEqual(len(chunks), 2)
        self.assertIn("First sentence. Second sentence! Third sentence?", chunks[0])
        self.assertIn("Fourth sentence. Fifth sentence. Sixth sentence.", chunks[1])

    def test_split_text_by_sentences_with_overlap(self):
        """Тест разделения текста по предложениям с перекрытием."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence. Fifth sentence."
        
        # Тест с 3 предложениями в чанке и 1 предложением перекрытия
        splitter = SentenceTextSplitter(sentences_per_chunk=3, sentence_overlap=1)
        chunks = splitter.split_text(text)
        
        # Ожидаем 2 чанка (из-за перекрытия):
        # 1. Первые 3 предложения
        # 2. 3-е, 4-е, 5-е предложения (с перекрытием)
        self.assertEqual(len(chunks), 2)
        self.assertIn("First sentence. Second sentence! Third sentence?", chunks[0])
        self.assertIn("Third sentence? Fourth sentence. Fifth sentence.", chunks[1])

    def test_split_text_empty(self):
        """Тест разделения пустого текста."""
        text = ""
        splitter = SentenceTextSplitter()
        chunks = splitter.split_text(text)
        self.assertEqual(chunks, [])

    def test_split_text_single_sentence(self):
        """Тест разделения текста с одним предложением."""
        text = "Single sentence."
        splitter = SentenceTextSplitter()
        chunks = splitter.split_text(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Single sentence.")

    def test_split_text_with_large_overlap(self):
        """Тест разделения текста с большим перекрытием."""
        text = "First sentence. Second sentence! Third sentence?"
        
        # Тест с 2 предложениями в чанке и 2 предложениями перекрытия (равно количеству предложений в чанке)
        splitter = SentenceTextSplitter(sentences_per_chunk=2, sentence_overlap=2)
        chunks = splitter.split_text(text)
        
        # Ожидаем 2 чанка (из-за механизма защиты от бесконечного цикла):
        # 1. Первые 2 предложения
        # 2. 2-е, 3-е предложения
        self.assertEqual(len(chunks), 2)
        self.assertIn("First sentence. Second sentence!", chunks[0])
        self.assertIn("Second sentence! Third sentence?", chunks[1])


if __name__ == '__main__':
    unittest.main()