"""Модуль для многоуровневого чанкинга и векторизации."""

import logging
from typing import Any, Dict, List, Union

from langchain_core.documents import Document

from core.indexing.chunker import get_text_splitter
from core.indexing.paragraph_chunker import ParagraphTextSplitter
from core.indexing.sentence_chunker import SentenceTextSplitter

logger = logging.getLogger(__name__)


class MultiLevelChunker:
    """Многоуровневый чанкер для создания иерархических чанков."""
    
    def __init__(
        self,
        macro_chunk_strategy: str = "character",
        macro_chunk_size: int = 10000,
        macro_chunk_overlap: int = 1000,
        macro_paragraphs_per_chunk: int = 5,
        macro_paragraph_overlap: int = 1,
        macro_sentences_per_chunk: int = 10,
        macro_sentence_overlap: int = 1,
        micro_chunk_strategy: str = "character",
        micro_chunk_size: int = 1000,
        micro_chunk_overlap: int = 100,
        micro_paragraphs_per_chunk: int = 3,
        micro_paragraph_overlap: int = 1,
        micro_sentences_per_chunk: int = 5,
        micro_sentence_overlap: int = 1
    ):
        """
        Инициализация многоуровневого чанкера с полностью настраиваемыми параметрами.
        
        Args:
            macro_chunk_strategy (str): Стратегия макро-чанкинга ("character", "paragraph", "sentence")
            macro_chunk_size (int): Размер макро-чанков в символах
            macro_chunk_overlap (int): Перекрытие макро-чанков в символах
            macro_paragraphs_per_chunk (int): Количество абзацев в макро-чанке
            macro_paragraph_overlap (int): Перекрытие абзацев в макро-чанках
            macro_sentences_per_chunk (int): Количество предложений в макро-чанке
            macro_sentence_overlap (int): Перекрытие предложений в макро-чанках
            micro_chunk_strategy (str): Стратегия микро-чанкинга ("character", "paragraph", "sentence")
            micro_chunk_size (int): Размер микро-чанков в символах
            micro_chunk_overlap (int): Перекрытие микро-чанков в символах
            micro_paragraphs_per_chunk (int): Количество абзацев в микро-чанке
            micro_paragraph_overlap (int): Перекрытие абзацев в микро-чанках
            micro_sentences_per_chunk (int): Количество предложений в микро-чанке
            micro_sentence_overlap (int): Перекрытие предложений в микро-чанках
        """
        self.macro_chunk_strategy = macro_chunk_strategy
        self.macro_chunk_size = macro_chunk_size
        self.macro_chunk_overlap = macro_chunk_overlap
        self.macro_paragraphs_per_chunk = macro_paragraphs_per_chunk
        self.macro_paragraph_overlap = macro_paragraph_overlap
        self.macro_sentences_per_chunk = macro_sentences_per_chunk
        self.macro_sentence_overlap = macro_sentence_overlap
        
        self.micro_chunk_strategy = micro_chunk_strategy
        self.micro_chunk_size = micro_chunk_size
        self.micro_chunk_overlap = micro_chunk_overlap
        self.micro_paragraphs_per_chunk = micro_paragraphs_per_chunk
        self.micro_paragraph_overlap = micro_paragraph_overlap
        self.micro_sentences_per_chunk = micro_sentences_per_chunk
        self.micro_sentence_overlap = micro_sentence_overlap
        
        # Создаем макро-чанкер
        if macro_chunk_strategy == "paragraph":
            self.macro_chunker = ParagraphTextSplitter(
                paragraphs_per_chunk=macro_paragraphs_per_chunk,
                paragraph_overlap=macro_paragraph_overlap
            )
        elif macro_chunk_strategy == "sentence":
            self.macro_chunker = SentenceTextSplitter(
                sentences_per_chunk=macro_sentences_per_chunk,
                sentence_overlap=macro_sentence_overlap
            )
        else:  # character
            self.macro_chunker = get_text_splitter(
                chunking_strategy="character",
                chunk_size=macro_chunk_size,
                chunk_overlap=macro_chunk_overlap
            )
        
        # Создаем микро-чанкер
        if micro_chunk_strategy == "paragraph":
            self.micro_chunker = ParagraphTextSplitter(
                paragraphs_per_chunk=micro_paragraphs_per_chunk,
                paragraph_overlap=micro_paragraph_overlap
            )
        elif micro_chunk_strategy == "sentence":
            self.micro_chunker = SentenceTextSplitter(
                sentences_per_chunk=micro_sentences_per_chunk,
                sentence_overlap=micro_sentence_overlap
            )
        else:  # character
            self.micro_chunker = get_text_splitter(
                chunking_strategy="character",
                chunk_size=micro_chunk_size,
                chunk_overlap=micro_chunk_overlap
            )
    
    def create_multilevel_chunks(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Создание многоуровневых чанков из документов.
        
        Args:
            documents (List[Document]): Список документов для чанкинга
            
        Returns:
            List[Dict[str, Any]]: Список многоуровневых чанков с метаданными
        """
        logger.info(f"Начало создания многоуровневых чанков из {len(documents)} документов")
        
        multilevel_chunks = []
        
        # Создаем макро-чанки
        macro_chunks = self.macro_chunker.split_documents(documents)
        
        logger.info(f"Создано {len(macro_chunks)} макро-чанков")
        
        for i, macro_chunk in enumerate(macro_chunks):
            # Создаем микро-чанки из макро-чанка
            micro_chunks = self.micro_chunker.split_text(macro_chunk.page_content)
            
            # Создаем запись для макро-чанка с ссылками на микро-чанки
            multilevel_chunk = {
                "macro_chunk": macro_chunk,
                "micro_chunks": micro_chunks,
                "chunk_index": i,
                "total_macro_chunks": len(macro_chunks),
                "total_micro_chunks": len(micro_chunks)
            }
            
            multilevel_chunks.append(multilevel_chunk)
            
            logger.debug(f"Макро-чанк {i+1}: {len(micro_chunks)} микро-чанков")
        
        logger.info(f"Всего создано {len(multilevel_chunks)} многоуровневых чанков")
        return multilevel_chunks
    
    def get_all_vectors_for_chunk(self, macro_chunk_content: str) -> List[str]:
        """
        Получение всех текстовых представлений для векторизации одного макро-чанка.
        
        Args:
            macro_chunk_content (str): Содержимое макро-чанка
            
        Returns:
            List[str]: Список текстовых представлений для векторизации
        """
        # Основной текст макро-чанка
        vectors = [macro_chunk_content]
        
        # Микро-чанки
        micro_chunks = self.micro_chunker.split_text(macro_chunk_content)
        vectors.extend(micro_chunks)
        
        return vectors


def create_multilevel_chunker_from_config(config: Dict[str, Any]) -> MultiLevelChunker:
    """
    Создание многоуровневого чанкера из конфигурации.
    
    Args:
        config (Dict[str, Any]): Конфигурация
        
    Returns:
        MultiLevelChunker: Экземпляр многоуровневого чанкера
    """
    return MultiLevelChunker(
        macro_chunk_strategy=config.get('multilevel_macro_strategy', 'character'),
        macro_chunk_size=config.get('multilevel_macro_chunk_size', 10000),
        macro_chunk_overlap=config.get('multilevel_macro_chunk_overlap', 1000),
        macro_paragraphs_per_chunk=config.get('multilevel_macro_paragraphs_per_chunk', 5),
        macro_paragraph_overlap=config.get('multilevel_macro_paragraph_overlap', 1),
        macro_sentences_per_chunk=config.get('multilevel_macro_sentences_per_chunk', 10),
        macro_sentence_overlap=config.get('multilevel_macro_sentence_overlap', 1),
        micro_chunk_strategy=config.get('multilevel_micro_strategy', 'character'),
        micro_chunk_size=config.get('multilevel_micro_chunk_size', 1000),
        micro_chunk_overlap=config.get('multilevel_micro_chunk_overlap', 100),
        micro_paragraphs_per_chunk=config.get('multilevel_micro_paragraphs_per_chunk', 3),
        micro_paragraph_overlap=config.get('multilevel_micro_paragraph_overlap', 1),
        micro_sentences_per_chunk=config.get('multilevel_micro_sentences_per_chunk', 5),
        micro_sentence_overlap=config.get('multilevel_micro_sentence_overlap', 1)
    )


def create_flexible_multilevel_chunker(
    macro_strategy: str = "character",
    macro_size: Union[int, str] = 10000,
    micro_strategy: str = "character",
    micro_size: Union[int, str] = 1000
) -> MultiLevelChunker:
    """
    Создание гибкого многоуровневого чанкера с простыми параметрами.
    
    Args:
        macro_strategy (str): Стратегия макро-чанкинга
        macro_size (Union[int, str]): Размер макро-чанков
        micro_strategy (str): Стратегия микро-чанкинга
        micro_size (Union[int, str]): Размер микро-чанков
        
    Returns:
        MultiLevelChunker: Экземпляр многоуровневого чанкера
    """
    # Определяем параметры на основе стратегий и размеров
    macro_params = {}
    micro_params = {}
    
    if macro_strategy == "character":
        macro_params = {
            "macro_chunk_strategy": "character",
            "macro_chunk_size": int(macro_size) if isinstance(macro_size, (int, float)) else 10000,
            "macro_chunk_overlap": int(int(macro_size) * 0.1) if isinstance(macro_size, (int, float)) else 1000
        }
    elif macro_strategy == "paragraph":
        macro_params = {
            "macro_chunk_strategy": "paragraph",
            "macro_paragraphs_per_chunk": int(macro_size) if isinstance(macro_size, (int, float)) else 5,
            "macro_paragraph_overlap": max(1, int(int(macro_size) * 0.2)) if isinstance(macro_size, (int, float)) else 1
        }
    elif macro_strategy == "sentence":
        macro_params = {
            "macro_chunk_strategy": "sentence",
            "macro_sentences_per_chunk": int(macro_size) if isinstance(macro_size, (int, float)) else 10,
            "macro_sentence_overlap": max(1, int(int(macro_size) * 0.1)) if isinstance(macro_size, (int, float)) else 1
        }
    
    if micro_strategy == "character":
        micro_params = {
            "micro_chunk_strategy": "character",
            "micro_chunk_size": int(micro_size) if isinstance(micro_size, (int, float)) else 1000,
            "micro_chunk_overlap": int(int(micro_size) * 0.1) if isinstance(micro_size, (int, float)) else 100
        }
    elif micro_strategy == "paragraph":
        micro_params = {
            "micro_chunk_strategy": "paragraph",
            "micro_paragraphs_per_chunk": int(micro_size) if isinstance(micro_size, (int, float)) else 3,
            "micro_paragraph_overlap": max(1, int(int(micro_size) * 0.3)) if isinstance(micro_size, (int, float)) else 1
        }
    elif micro_strategy == "sentence":
        micro_params = {
            "micro_chunk_strategy": "sentence",
            "micro_sentences_per_chunk": int(micro_size) if isinstance(micro_size, (int, float)) else 5,
            "micro_sentence_overlap": max(1, int(int(micro_size) * 0.2)) if isinstance(micro_size, (int, float)) else 1
        }
    
    # Объединяем параметры
    params = {**macro_params, **micro_params}
    
    return MultiLevelChunker(**params)