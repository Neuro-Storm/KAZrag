#!/usr/bin/env python3
"""Скрипт для оценки различных моделей эмбеддингов."""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Добавляем путь к проекту в sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Импортируем модули проекта
from config.config_manager import ConfigManager
from config.settings import Config
from core.embedding.embedding_manager import EmbeddingManager
from core.indexing.indexer import run_indexing_logic
from core.search.searcher import search_in_collection
from core.qdrant.qdrant_client import get_qdrant_client
from langchain_core.documents import Document  # Добавляем импорт Document

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Список моделей для тестирования
# Включаем различные модели для оценки
EMBEDDING_MODELS_TO_TEST = [
    # ==== SBERT / RuBERT / Russian-specific ====
    ##"ai-forever/sbert_large_nlu_ru",        # 2020, ~427M, MIT, русский
    ##"ai-forever/sbert_large_mt_nlu_ru",     # 2023, ~427M, MIT, русский+multilingual
    ##"ai-forever/ru-en-RoSBERTa",            # 2024, ~404M, MIT, русский+английский (SOTA на ruMTEB)
    ##"ai-forever/ruBert-base",               # 2020, ~172M, MIT, русский
    ##"ai-forever/ruBert-large",              # 2020, ~329M, MIT, русский
    ###"ai-forever/ruRoberta-base",            # 2021, ~125M, MIT, русский НЕ РАБОТАЕТ
    ###"ai-forever/ruRoberta-large",           # 2021, ~355M, MIT, русский НЕ РАБОТАЕТ
    ###"ai-forever/rubert-tiny2",              # 2021, ~59M, Apache-2.0, русский НЕ РАБОТАЕТ
    ###"cointegrated/rubert-tiny-sentence",    # 2021, ~29M, MIT, русский (sentence-level) НЕ РАБОТАЕТ
    ##"cointegrated/rubert-tiny2",            # 2021, ~29M, MIT, русский (альтернатива)
    ##"DeepPavlov/rubert-base-cased",         # 2019, ~177M, MIT, русский
    ##"DeepPavlov/rubert-base-cased-sentence",# 2019, ~180M, MIT, русский (sentence-level)
    ##"DeepPavlov/rubert-base-cased-conversational", # 2020, ~177M, MIT, русский (диалоги)

    # ==== FRED-T5 (русские encoder-decoder модели) ====
    ###"ai-forever/FRED-T5-small",             # 2023, ~60M, MIT, русский НЕ РАБОТАЕТ
    ###"ai-forever/FRED-T5-base",              # 2023, ~245M, MIT, русский НЕ РАБОТАЕТ
    ###"ai-forever/FRED-T5-large",             # 2023, ~770M, MIT, русский НЕ РАБОТАЕТ
    ###"ai-forever/FRED-T5-1.7B",              # 2023, ~1.7B, MIT, русский (FRIDA проект) НЕ РАБОТАЕТ

    # ==== ruGPT3 (можно использовать для embeddings через pooling) ====
    ##"sberbank-ai/rugpt3small_based_on_gpt2",   # 2021, ~117M, SberBank Open License, русский
    ##"sberbank-ai/rugpt3medium_based_on_gpt2",  # 2021, ~770M, SberBank Open License, русский
    ###"sberbank-ai/rugpt3large_based_on_gpt2",   # 2021, ~1.3B, SberBank Open License, русский НЕ РАБОТАЕТ

    # ==== USER / Russian-focused modern ====
    ##"deepvk/USER-base",                     # 2022, ~110M, Apache-2.0, русский (Encodechka/MTEB топ)
    ##"deepvk/USER-bge-m3",                   # 2023, ~1.3B, Apache-2.0, мультиязычная (incl. RU)

    # ==== Multilingual SBERT family ====
    ##"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", # 2020, ~118M, Apache-2.0, multilingual incl. RU
    ##"sentence-transformers/paraphrase-multilingual-mpnet-base-v2", # 2021, ~177M, Apache-2.0, multilingual incl. RU
    ##"sentence-transformers/paraphrase-xlm-r-multilingual-v1",      # 2020, ~278M, Apache-2.0, multilingual incl. RU
    ##"sentence-transformers/distiluse-base-multilingual-cased-v1",  # 2020, ~134M, Apache-2.0, multilingual incl. RU
    ##"sentence-transformers/distiluse-base-multilingual-cased-v2",  # 2020, ~134M, Apache-2.0, multilingual incl. RU
    ##"sentence-transformers/LaBSE",          # 2020, ~470M, Apache-2.0, multilingual incl. RU
    ##"cointegrated/LaBSE-en-ru",             # 2020, ~129M, Apache-2.0, RU+EN адаптация

    # ==== E5 family ====
    ##"intfloat/multilingual-e5-small",       # 2023, ~110M, MIT, multilingual incl. RU
    ##"intfloat/multilingual-e5-base",        # 2023, ~355M, MIT, multilingual incl. RU
    ##"intfloat/multilingual-e5-large",       # 2023, ~770M, MIT, multilingual incl. RU
    ##"intfloat/e5-base-v2",                  # 2023, ~137M, MIT, multilingual incl. RU
    ##"intfloat/e5-large-v2",                 # 2023, ~355M, MIT, multilingual incl. RU
    "d0rj/e5-base-en-ru",                   # 2023, ~132M, MIT, RU+EN
    ##"hivaze/ru-e5-base",                    # 2023, ~139M, Apache-2.0, RU+EN

    # ==== BGE family (BAAI) ====
    ##"BAAI/bge-m3",                          # 2024, ~567M / 1.3B, MIT, multilingual incl. RU
    #"BAAI/bge-multilingual-gemma2",         # 2024, ~9B, MIT, multilingual incl. RU

    # ==== GTE (Alibaba) ====
    ###"Alibaba-NLP/gte-multilingual-base",    # 2024, ~305M, Apache-2.0, multilingual incl. RU НЕ РАБОТАЕТ
    #"Alibaba-NLP/gte-Qwen1.5-7B-instruct",  # 2023, ~7B, Apache-2.0, multilingual incl. RU

    # ==== Qwen Embedding family ====
    "Qwen/Qwen3-Embedding-0.6B",            # 2025, 0.6B, Apache-2.0, multilingual incl. RU
    #"Qwen/Qwen3-Embedding-4B",              # 2025, 4B, Apache-2.0, multilingual incl. RU
    #"Qwen/Qwen3-Embedding-8B",              # 2025, 8B, Apache-2.0, multilingual incl. RU
    #"Qwen/Qwen3-Embedding-0.6B-GGUF",       # 2025, 0.6B quantized, Apache-2.0, multilingual incl. RU
    #"models/Qwen3-Embedding-4B-Q8_0.gguf",   # 2025, 4B quantized, Apache-2.0, multilingual incl. RU

    # ==== QZhou (новый SOTA на базе Qwen2.5) ====
    ###"QZhou/QZhou-Embedding",                # 2025, основан на Qwen2.5-7B, Apache-2.0, SOTA на MTEB/CMTEB incl. RU НЕ РАБОТАЕТ

    # ==== Classical multilingual models (можно использовать для embeddings) ====
    "bert-base-multilingual-cased",         # 2019, ~178M, Apache-2.0, multilingual incl. RU
    "FacebookAI/xlm-roberta-base",          # 2019, ~270M, MIT, multilingual incl. RU
    "FacebookAI/xlm-roberta-large",         # 2019, ~585M, MIT, multilingual incl. RU
    "google/mt5-base",                      # 2020, ~580M, Apache-2.0, multilingual incl. RU
    "google/mt5-large",                     # 2020, ~1.2B, Apache-2.0, multilingual incl. RU
    #"google/mt5-xl",                        # 2020, ~11B, Apache-2.0, multilingual incl. RU

    # ==== Classical embeddings (word-level) ====
    ###"facebookresearch/fastText",            # 2017–2018, 300D, CC BY-SA, multilingual incl. RU НЕ РАБОТАЕТ
    ###"facebookresearch/LASER",               # 2019–2022, 1024D, BSD, multilingual incl. RU НЕ РАБОТАЕТ
    ###"natasha/navec",                        # 2019, ~1.9M vocab, MIT, русский НЕ РАБОТАЕТ

]

@dataclass
class EvaluationResult:
    """Результат оценки модели."""
    model_name: str
    collection_name: str
    indexing_time: float
    search_time: float
    search_results: List[Dict]
    test_queries: List[str]
    scores: Dict[str, float]
    timestamp: str

class EmbeddingEvaluator:
    """Класс для оценки моделей эмбеддингов."""
    
    def __init__(self, test_data_dir: str = "./test_data", results_dir: str = "./evaluation_results"):
        """Инициализация оценщика."""
        self.test_data_dir = Path(test_data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Создаем директорию для тестовых данных
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Получаем менеджер конфигурации
        self.config_manager = ConfigManager.get_instance()
        self.embedding_manager = EmbeddingManager.get_instance()
        
    def prepare_test_data(self):
        """Подготовка тестовых данных."""
        logger.info("Подготовка тестовых данных...")
        logger.info("Используется существующая папка data_to_index1")
        
    async def evaluate_model(self, model_name: str, pre_chunked_documents=None) -> EvaluationResult:
        """Оценка конкретной модели эмбеддингов."""
        logger.info(f"Оценка модели: {model_name}")
        
        # Получаем текущую конфигурацию
        config = self.config_manager.get()
        
        # Сохраняем оригинальные настройки
        original_model = config.current_hf_model
        original_collection = config.collection_name
        original_folder = config.folder_path
        
        try:
            # Обновляем конфигурацию для тестирования
            config.current_hf_model = model_name
            config.collection_name = f"test_collection_{model_name.replace('/', '_').replace('-', '_')}"
            # Используем папку data_to_index1 вместо тестовой папки
            config.folder_path = "./data_to_index1"
            config.force_recreate = True
            
            # Если модель требует токен HuggingFace, устанавливаем его из переменных окружения
            # Проверяем, есть ли в названии модели признаки приватной модели
            if "private" in model_name.lower() or "protected" in model_name.lower():
                import os
                hf_token = os.getenv("HUGGINGFACE_TOKEN")
                if hf_token:
                    config.huggingface_token = hf_token
                else:
                    logger.warning(f"Модель {model_name} может требовать токен HuggingFace, но он не найден в переменных окружения")
            
            # Сохраняем изменения в конфигурации
            self.config_manager.save(config)
            
            # Очищаем кэш эмбеддеров
            self.embedding_manager.clear_cache()
            
            # Принудительно очищаем память PyTorch
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            # Тестовые запросы, соответствующие тематике документов в data_to_index1
            test_queries = [
                "Как влияют электростатические поля на теплоотдачу в моторных авиационных маслах?",
                "Какова плотность масла МС-20 при различных температурах и давлениях?",
               # "Какие проблемы возникают в системах смазки авиационных двигателей?",
               # "Какие методы используются для расчета теплоотдачи к моторным маслам?",
                "Как происходит осадкообразование в системах смазки двигателей?"
            ]
            
            # Запуск индексации
            logger.info("Запуск индексации...")
            start_time = datetime.now()
            indexing_success, indexing_status = await run_indexing_logic(pre_chunked_documents=pre_chunked_documents)
            indexing_time = (datetime.now() - start_time).total_seconds()
            
            if not indexing_success:
                logger.error(f"Индексация не удалась для модели {model_name}: {indexing_status}")
                return EvaluationResult(
                    model_name=model_name,
                    collection_name=config.collection_name,
                    indexing_time=indexing_time,
                    search_time=0,
                    search_results=[],
                    test_queries=test_queries,
                    scores={"error": "indexing_failed"},
                    timestamp=datetime.now().isoformat()
                )
            
            logger.info(f"Индексация завершена за {indexing_time:.2f} секунд")
            
            # Выполняем поиск по тестовым запросам
            logger.info("Выполнение поиска...")
            search_results = []
            total_search_time = 0
            
            # Получаем клиент Qdrant
            client = get_qdrant_client(config)
            
            for query in test_queries:
                start_time = datetime.now()
                results, error = await search_in_collection(
                    query=query,
                    collection_name=config.collection_name,
                    device="cpu",  # Используем CPU для тестирования
                    k=3,  # Получаем 3 лучших результата
                    client=client
                )
                search_time = (datetime.now() - start_time).total_seconds()
                total_search_time += search_time
                
                if error:
                    logger.warning(f"Ошибка поиска для запроса '{query}': {error}")
                    search_results.append({
                        "query": query,
                        "results": [],
                        "error": error,
                        "search_time": search_time
                    })
                else:
                    processed_results = []
                    for result, score in results:
                        processed_results.append({
                            "content": result.get("content", ""),  # Сохраняем полное содержимое
                            "score": score,
                            "metadata": result.get("metadata", {})
                        })
                    search_results.append({
                        "query": query,
                        "results": processed_results,
                        "search_time": search_time
                    })
            
            avg_search_time = total_search_time / len(test_queries) if test_queries else 0
            logger.info(f"Поиск завершен. Среднее время: {avg_search_time:.4f} секунд")
            
            # Вычисляем базовые метрики
            scores = self._calculate_basic_scores(search_results)
            
            # Создаем результат оценки
            result = EvaluationResult(
                model_name=model_name,
                collection_name=config.collection_name,
                indexing_time=indexing_time,
                search_time=avg_search_time,
                search_results=search_results,
                test_queries=test_queries,
                scores=scores,
                timestamp=datetime.now().isoformat()
            )
            
            return result
            
        except Exception as e:
            logger.exception(f"Ошибка при оценке модели {model_name}: {e}")
            return EvaluationResult(
                model_name=model_name,
                collection_name="",
                indexing_time=0,
                search_time=0,
                search_results=[],
                test_queries=test_queries,
                scores={"error": str(e)},
                timestamp=datetime.now().isoformat()
            )
        finally:
            # Восстанавливаем оригинальную конфигурацию
            config.current_hf_model = original_model
            config.collection_name = original_collection
            config.folder_path = original_folder
            config.force_recreate = False
            self.config_manager.save(config)
            
            # Очищаем кэш
            self.embedding_manager.clear_cache()
            
            # Принудительно очищаем память PyTorch
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    
    def _calculate_basic_scores(self, search_results: List[Dict]) -> Dict[str, float]:
        """Вычисление базовых метрик оценки."""
        if not search_results:
            return {"total_results": 0}
            
        total_results = sum(len(sr.get("results", [])) for sr in search_results)
        avg_results_per_query = total_results / len(search_results)
        
        # Подсчитываем количество результатов с высокой релевантностью (score > 0.5)
        high_relevance_count = 0
        total_score = 0
        score_count = 0
        
        for sr in search_results:
            for result in sr.get("results", []):
                score = result.get("score", 0)
                total_score += score
                score_count += 1
                if score > 0.5:
                    high_relevance_count += 1
                    
        avg_score = total_score / score_count if score_count > 0 else 0
        relevance_ratio = high_relevance_count / total_results if total_results > 0 else 0
        
        return {
            "total_results": total_results,
            "avg_results_per_query": avg_results_per_query,
            "avg_score": avg_score,
            "high_relevance_ratio": relevance_ratio
        }
    
    async def run_evaluation(self) -> List[EvaluationResult]:
        """Запуск оценки всех моделей."""
        logger.info("Запуск оценки моделей эмбеддингов")
        
        # Подготавливаем тестовые данные
        self.prepare_test_data()
        
        # Получаем текущую конфигурацию для отображения настроек чанкования
        config = self.config_manager.get()
        chunk_settings_hash = self._get_chunk_settings_hash(config)
        logger.info(f"Текущие настройки чанкования (хэш: {chunk_settings_hash}):")
        logger.info(f"  Стратегия: {config.chunking_strategy}")
        logger.info(f"  Размер чанка: {config.chunk_size}")
        logger.info(f"  Перекрытие: {config.chunk_overlap}")
        logger.info(f"  Абзацев в чанке: {config.paragraphs_per_chunk}")
        logger.info(f"  Перекрытие абзацев: {config.paragraph_overlap}")
        
        # ВСЕГДА выполняем предварительное чанкование при запуске скрипта
        # Это гарантирует, что используются актуальные настройки чанкования
        logger.info("Выполняем ПРИНУДИТЕЛЬНОЕ чанкование документов...")
        pre_chunked_documents = await self._get_pre_chunked_documents()
        logger.info(f"Чанкование завершено. Создано чанков: {len(pre_chunked_documents)}")
        
        # Список для хранения результатов
        results = []
        
        # Оцениваем каждую модель
        for model_name in EMBEDDING_MODELS_TO_TEST:
            try:
                # Получаем информацию о кэше перед оценкой
                cache_info_before = self.embedding_manager.get_cache_info()
                logger.info(f"Кэш перед оценкой модели {model_name}: {cache_info_before}")
                
                result = await self.evaluate_model(model_name, pre_chunked_documents)
                results.append(result)
                
                # Сохраняем промежуточный результат
                self._save_result(result)
                
                logger.info(f"Оценка модели {model_name} завершена")
                logger.info(f"  Время индексации: {result.indexing_time:.2f} сек")
                logger.info(f"  Среднее время поиска: {result.search_time:.4f} сек")
                logger.info(f"  Общее количество результатов: {result.scores.get('total_results', 0)}")
                logger.info(f"  Средний score: {result.scores.get('avg_score', 0):.4f}")
                
                # Получаем информацию о кэше после оценки
                cache_info_after = self.embedding_manager.get_cache_info()
                logger.info(f"Кэш после оценки модели {model_name}: {cache_info_after}")
                
                # Явно очищаем кэш между оценками моделей
                self.embedding_manager.clear_cache()
                logger.info(f"Кэш очищен после оценки модели {model_name}")
            except Exception as e:
                logger.exception(f"Ошибка при оценке модели {model_name}: {e}")
                # В случае ошибки тоже очищаем кэш
                self.embedding_manager.clear_cache()
                logger.info(f"Кэш очищен после ошибки при оценке модели {model_name}")
                continue
        
        # Сохраняем все результаты
        self._save_all_results(results)
        
        return results
    
    def _save_result(self, result: EvaluationResult):
        """Сохранение результата оценки отдельной модели."""
        result_file = self.results_dir / f"evaluation_{result.model_name.replace('/', '_').replace('-', '_')}_{result.timestamp.replace(':', '-')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)
        logger.info(f"Результат сохранен в {result_file}")
    
    def _save_all_results(self, results: List[EvaluationResult]):
        """Сохранение всех результатов в одном файле."""
        summary_file = self.results_dir / f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_data = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_models_evaluated": len(results),
            "results": [asdict(result) for result in results]
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Сводка результатов сохранена в {summary_file}")
        
        # Создаем читаемый отчет
        self._generate_readable_report(results, summary_file.with_suffix('.txt'))
    
    async def _get_pre_chunked_documents(self):
        """Получение предварительно нарезанных документов с использованием стандартного механизма.
        ВСЕГДА создает новые чанки, игнорируя любое кэширование."""
        logger.info("Получение предварительно нарезанных документов (ВСЕГДА создаем новые)...")
        
        try:
            # Получаем текущую конфигурацию
            config = self.config_manager.get()
            
            # Проверяем настройки чанкования
            chunk_settings_hash = self._get_chunk_settings_hash(config)
            logger.info(f"Хэш настроек чанкования: {chunk_settings_hash}")
            logger.info(f"Настройки чанкования: strategy={config.chunking_strategy}, "
                       f"size={config.chunk_size}, overlap={config.chunk_overlap}, "
                       f"paragraphs_per_chunk={config.paragraphs_per_chunk}")
            
            # Сохраняем оригинальные настройки
            original_folder = config.folder_path
            original_force_recreate = config.force_recreate
            
            try:
                # Временно изменяем настройки для получения нарезанных документов
                # Используем папку data_to_index1
                config.folder_path = "./data_to_index1"
                config.force_recreate = True  # ВСЕГДА пересоздаем для гарантии актуальности
                
                # Сохраняем изменения в конфигурации
                self.config_manager.save(config)
                
                # Импортируем необходимые модули
                from core.indexing.document_loader import DocumentLoader
                from core.indexing.text_splitter import TextSplitter
                from pathlib import Path
                
                # Создаем компоненты для нарезки
                document_loader = DocumentLoader()
                text_splitter = TextSplitter(config)
                
                folder_path = Path(config.folder_path)
                folder_path_resolved = folder_path.resolve()
                
                # Собираем все документы и нарезаем их
                all_documents = []
                
                # Обрабатываем .txt файлы
                txt_files = list(folder_path.rglob("*.txt"))
                logger.info(f"Найдено .txt файлов: {len(txt_files)}")
                for filepath in txt_files:
                    try:
                        logger.info(f"Загрузка файла: {filepath}")
                        loaded_docs = document_loader.load_text_file(filepath)
                        logger.info(f"Загружено документов из {filepath}: {len(loaded_docs)}")
                        
                        if loaded_docs:
                            logger.info(f"Первый документ до нарезки: {loaded_docs[0].page_content[:100]}...")
                            
                        chunks = text_splitter.split_documents(loaded_docs)
                        logger.info(f"Создано чанков из {filepath}: {len(chunks)}")
                        
                        if chunks:
                            logger.info(f"Первый чанк: {chunks[0].page_content[:100]}...")
                            logger.info(f"Метаданные первого чанка: {chunks[0].metadata}")
                        
                        processed_chunks = self._process_chunks_for_pre_chunking(chunks, filepath, folder_path_resolved, config)
                        all_documents.extend(processed_chunks)
                    except Exception as e:
                        logger.exception(f"Ошибка при обработке файла {filepath}: {e}")
                        continue
                
                # Обрабатываем .md файлы
                md_files = list(folder_path.rglob("*.md"))
                logger.info(f"Найдено .md файлов: {len(md_files)}")
                for filepath in md_files:
                    try:
                        logger.info(f"Загрузка файла: {filepath}")
                        loaded_docs = document_loader.load_text_file(filepath)
                        logger.info(f"Загружено документов из {filepath}: {len(loaded_docs)}")
                        
                        if loaded_docs:
                            logger.info(f"Первый документ до нарезки: {loaded_docs[0].page_content[:100]}...")
                            
                        chunks = text_splitter.split_documents(loaded_docs)
                        logger.info(f"Создано чанков из {filepath}: {len(chunks)}")
                        
                        if chunks:
                            logger.info(f"Первый чанк: {chunks[0].page_content[:100]}...")
                            logger.info(f"Метаданные первого чанка: {chunks[0].metadata}")
                        
                        processed_chunks = self._process_chunks_for_pre_chunking(chunks, filepath, folder_path_resolved, config)
                        all_documents.extend(processed_chunks)
                    except Exception as e:
                        logger.exception(f"Ошибка при обработке файла {filepath}: {e}")
                        continue
                
                logger.info(f"Предварительная нарезка завершена. Всего чанков: {len(all_documents)}")
                if all_documents:
                    logger.info(f"Примеры чанков:")
                    for i, doc in enumerate(all_documents[:3]):
                        logger.info(f"  Чанк {i+1}: {doc.page_content[:100]}...")
                return all_documents
                
            finally:
                # Восстанавливаем оригинальную конфигурацию
                config.folder_path = original_folder
                config.force_recreate = original_force_recreate
                self.config_manager.save(config)
                
        except Exception as e:
            logger.exception(f"Ошибка при получении предварительно нарезанных документов: {e}")
            return []
    
    def _get_chunk_settings_hash(self, config: Config) -> str:
        """Создание хэша настроек чанкования для проверки их изменений."""
        import hashlib
        
        # Собираем все настройки чанкования в строку
        chunk_settings = (
            f"{config.chunking_strategy}_"
            f"{config.chunk_size}_"
            f"{config.chunk_overlap}_"
            f"{config.paragraphs_per_chunk}_"
            f"{config.paragraph_overlap}_"
            f"{config.sentences_per_chunk}_"
            f"{config.sentence_overlap}"
        )
        
        # Создаем хэш
        return hashlib.md5(chunk_settings.encode('utf-8')).hexdigest()
    
    def _process_chunks_for_pre_chunking(self, chunks: List[Document], filepath: Path, folder_path_resolved: Path, config: Config) -> List[Document]:
        """
        Обработка чанков для предварительной нарезки.
        По сути, это копия _process_chunks, но в контексте этого класса.
        """
        from core.indexing.metadata_manager import metadata_manager
        
        try:
            # Получение относительного пути файла от корневой папки
            abs_filepath = filepath.resolve()
            relative_source_path = abs_filepath.relative_to(folder_path_resolved)
        except ValueError:
            # Если файл не в корневой папке, используем только имя файла
            logger.warning(f"Файл {filepath} не находится внутри {folder_path_resolved}. Используется только имя файла.")
            relative_source_path = abs_filepath.name
        
        # Добавляем метаданные к чанкам с помощью MetadataManager
        # Учитываем настройки из конфигурации
        if config.enable_metadata_extraction:
            for chunk in chunks:
                # Добавляем пользовательские поля из конфигурации
                chunk = metadata_manager.add_metadata_to_chunk(chunk, filepath, config.metadata_custom_fields)
                # Обновляем source в метаданных для совместимости
                chunk.metadata["source"] = str(relative_source_path)
        else:
            # Если извлечение метаданных отключено, добавляем только базовые метаданные
            for chunk in chunks:
                chunk.metadata["source"] = str(relative_source_path)
        
        return chunks
    
    def _generate_readable_report(self, results: List[EvaluationResult], report_file: Path):
        """Генерация читаемого отчета."""
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("ОТЧЕТ ОЦЕНКИ МОДЕЛЕЙ ЭМБЕДДИНГОВ\r\n")
            f.write("=" * 50 + "\r\n")
            f.write(f"Дата оценки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\r\n")
            f.write(f"Количество оцененных моделей: {len(results)}\r\n\r\n")
            
            # Сортируем результаты по среднему score
            sorted_results = sorted(results, key=lambda x: x.scores.get('avg_score', 0), reverse=True)
            
            for i, result in enumerate(sorted_results, 1):
                f.write(f"{i}. МОДЕЛЬ: {result.model_name}\r\n")
                f.write("-" * 30 + "\r\n")
                f.write(f"   Время индексации: {result.indexing_time:.2f} сек\r\n")
                f.write(f"   Среднее время поиска: {result.search_time:.4f} сек\r\n")
                f.write(f"   Общее количество результатов: {result.scores.get('total_results', 0)}\r\n")
                f.write(f"   Среднее количество результатов на запрос: {result.scores.get('avg_results_per_query', 0):.2f}\r\n")
                f.write(f"   Средний score: {result.scores.get('avg_score', 0):.4f}\r\n")
                f.write(f"   Доля высокорелевантных результатов: {result.scores.get('high_relevance_ratio', 0):.2%}\r\n")
                f.write("\r\n")
                
                # Примеры результатов поиска
                f.write("   Примеры поиска:\r\n")
                for search_result in result.search_results[:2]:  # Показываем первые 2 запроса
                    f.write(f"     Запрос: {search_result['query']}\r\n")
                    f.write(f"     Время поиска: {search_result['search_time']:.4f} сек\r\n")
                    for j, res in enumerate(search_result['results'][:2], 1):  # Показываем первые 2 результата
                        f.write(f"       {j}. Score: {res['score']:.4f} - {res['content'][:100]}...\r\n")
                    f.write("\r\n")
                f.write("\r\n")
        
        logger.info(f"Читаемый отчет сохранен в {report_file}")

async def main():
    """Основная функция."""
    logger.info("Запуск оценки моделей эмбеддингов")
    
    # Создаем оценщик
    evaluator = EmbeddingEvaluator()
    
    # Запускаем оценку
    results = await evaluator.run_evaluation()
    
    # Выводим сводку
    logger.info("Оценка завершена. Сводка результатов:")
    sorted_results = sorted(results, key=lambda x: x.scores.get('avg_score', 0), reverse=True)
    
    print("\nРЕЙТИНГ МОДЕЛЕЙ ПО СРЕДНЕМУ SCORE:")
    print("-" * 50)
    for i, result in enumerate(sorted_results, 1):
        print(f"{i}. {result.model_name}")
        print(f"   Средний score: {result.scores.get('avg_score', 0):.4f}")
        print(f"   Время индексации: {result.indexing_time:.2f} сек")
        print(f"   Среднее время поиска: {result.search_time:.4f} сек")
        print()

if __name__ == "__main__":
    asyncio.run(main())