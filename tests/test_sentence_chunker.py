"""Тесты для проверки чанкинга по предложениям."""

import pytest

from core.indexing.sentence_chunker import SentenceTextSplitter


class TestSentenceChunker:
    """Тесты для SentenceTextSplitter."""

    def test_split_text_by_sentences(self):
        """Тест разделения текста по предложениям."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence. Fifth sentence."
        
        # Тест с настройками по умолчанию (5 предложений в чанке, 1 предложение перекрытия)
        splitter = SentenceTextSplitter()
        chunks = splitter.split_text(text)
        
        # Ожидаем 1 чанк, так как у нас всего 5 предложений
        assert len(chunks) == 1
        # Проверяем, что чанк содержит все предложения
        assert "First sentence." in chunks[0]
        assert "Second sentence!" in chunks[0]
        assert "Third sentence?" in chunks[0]
        assert "Fourth sentence." in chunks[0]
        assert "Fifth sentence." in chunks[0]

    def test_split_text_by_sentences_custom_settings(self):
        """Тест разделения текста по предложениям с пользовательскими настройками."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence. Fifth sentence. Sixth sentence."
        
        # Тест с 3 предложениями в чанке и без перекрытия
        splitter = SentenceTextSplitter(sentences_per_chunk=3, sentence_overlap=0)
        chunks = splitter.split_text(text)
        
        # Ожидаем 2 чанка:
        # 1. Первые 3 предложения
        # 2. Последние 3 предложения
        assert len(chunks) == 2
        assert "First sentence. Second sentence! Third sentence?" in chunks[0]
        assert "Fourth sentence. Fifth sentence. Sixth sentence." in chunks[1]

    def test_split_text_by_sentences_with_overlap(self):
        """Тест разделения текста по предложениям с перекрытием."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence. Fifth sentence."
        
        # Тест с 3 предложениями в чанке и 1 предложением перекрытия
        splitter = SentenceTextSplitter(sentences_per_chunk=3, sentence_overlap=1)
        chunks = splitter.split_text(text)
        
        # Ожидаем 2 чанка (из-за перекрытия):
        # 1. Первые 3 предложения
        # 2. 3-е, 4-е, 5-е предложения (с перекрытием)
        assert len(chunks) == 2
        assert "First sentence. Second sentence! Third sentence?" in chunks[0]
        assert "Third sentence? Fourth sentence. Fifth sentence." in chunks[1]

    def test_split_text_empty(self):
        """Тест разделения пустого текста."""
        text = ""
        splitter = SentenceTextSplitter()
        chunks = splitter.split_text(text)
        assert chunks == []

    def test_split_text_single_sentence(self):
        """Тест разделения текста с одним предложением."""
        text = "Single sentence."
        splitter = SentenceTextSplitter()
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == "Single sentence."

    def test_split_text_with_large_overlap(self):
        """Тест разделения текста с большим перекрытием."""
        text = "First sentence. Second sentence! Third sentence?"
        
        # Тест с 2 предложениями в чанке и 2 предложениями перекрытия (равно количеству предложений в чанке)
        splitter = SentenceTextSplitter(sentences_per_chunk=2, sentence_overlap=2)
        chunks = splitter.split_text(text)
        
        # Ожидаем 2 чанка (из-за механизма защиты от бесконечного цикла):
        # 1. Первые 2 предложения
        # 2. 2-е, 3-е предложения
        assert len(chunks) == 2
        assert "First sentence. Second sentence!" in chunks[0]
        assert "Second sentence! Third sentence?" in chunks[1]

    def test_split_text_with_zero_sentences_per_chunk(self):
        """Тест разделения текста с нулевым количеством предложений в чанке."""
        text = "First sentence. Second sentence."
        
        # Тест с 0 предложений в чанке
        splitter = SentenceTextSplitter(sentences_per_chunk=0)
        chunks = splitter.split_text(text)
        
        # Ожидаем 1 чанк с всем текстом
        assert len(chunks) == 1
        assert "First sentence. Second sentence." in chunks[0]

    def test_split_text_with_negative_overlap(self):
        """Тест разделения текста с отрицательным перекрытием."""
        text = "First sentence. Second sentence! Third sentence?"
        
        # Тест с отрицательным перекрытием
        splitter = SentenceTextSplitter(sentences_per_chunk=2, sentence_overlap=-1)
        chunks = splitter.split_text(text)
        
        # Ожидаем 2 чанка без перекрытия (отрицательное значение должно быть обработано)
        assert len(chunks) == 2
        assert "First sentence. Second sentence!" in chunks[0]
        assert "Third sentence?" in chunks[1]

    def test_split_text_with_excessive_overlap(self):
        """Тест разделения текста с чрезмерным перекрытием."""
        text = "First sentence. Second sentence! Third sentence."
        
        # Тест с перекрытием больше, чем количество предложений в чанке
        splitter = SentenceTextSplitter(sentences_per_chunk=2, sentence_overlap=5)
        chunks = splitter.split_text(text)
        
        # Ожидаем 2 чанка (перекрытие должно быть ограничено количеством предложений в чанке)
        assert len(chunks) == 2
        assert "First sentence. Second sentence!" in chunks[0]
        assert "Second sentence! Third sentence." in chunks[1]

    @pytest.mark.parametrize("sentences_per_chunk,sentence_overlap,expected_chunks", [
        (1, 0, 5),  # Одно предложение в чанке без перекрытия
        (5, 0, 1),  # Все предложения в одном чанке
        (3, 2, 2),  # Три предложения в чанке с двумя перекрытиями
        (2, 1, 4),  # Два предложения в чанке с одним перекрытием
    ])
    def test_split_text_parametrized(self, sentences_per_chunk, sentence_overlap, expected_chunks):
        """Параметризованный тест разделения текста по предложениям."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence. Fifth sentence."
        
        splitter = SentenceTextSplitter(
            sentences_per_chunk=sentences_per_chunk,
            sentence_overlap=sentence_overlap
        )
        chunks = splitter.split_text(text)
        assert len(chunks) == expected_chunks