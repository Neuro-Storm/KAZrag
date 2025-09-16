#!/usr/bin/env python3
"""Скрипт для детального тестирования производительности асинхронного поиска."""

import asyncio
import time
import logging
import statistics
from typing import List, Tuple, Any
from config.config_manager import ConfigManager
from core.search.searcher import search_in_collection
from core.qdrant.qdrant_client import aget_qdrant_client

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Расширенный список тестовых запросов
TEST_QUERIES = [
    "Как влияют электростатические поля на теплоотдачу в моторных авиационных маслах?",
    "Какова плотность масла МС-20 при различных температурах и давлениях?",
    "Какие проблемы возникают в системах смазки авиационных двигателей?",
    "Какие методы используются для расчета теплоотдачи к моторным маслам?",
    "Как происходит осадкообразование в системах смазки двигателей?",
    "Какие свойства имеют синтетические масла по сравнению с минеральными?",
    "Какие технологии применяются для очистки моторных масел?",
    "Как влияет вязкость масла на работу двигателя?",
    "Какие добавки используются в современных моторных маслах?",
    "Какие стандарты существуют для авиационных масел?",
    "Каковы характеристики термической стабильности авиационных масел?",
    "Какие методы используются для определения вязкости масел?",
    "Как влияет температура на свойства моторных масел?",
    "Какие требования предъявляются к маслам для реактивных двигателей?",
    "Как производится контроль качества авиационных масел?"
]

class AsyncSearchTester:
    """Класс для тестирования асинхронного поиска."""
    
    def __init__(self):
        self.config_manager = ConfigManager.get_instance()
        self.config = self.config_manager.get()
        self.collection_name = self.config.collection_name
        self.search_device = "cuda" if self.config.device == "cuda" else "cpu"
        self.k = self.config.search_default_k
        
    async def single_search(self, query: str, query_id: int) -> Tuple[int, str, float, int, float]:
        """Выполняет один поисковый запрос и возвращает метрики."""
        start_time = time.time()
        start_perf_counter = time.perf_counter()
        
        try:
            # Получаем клиента Qdrant
            client = await aget_qdrant_client(self.config)
            
            # Выполняем поиск
            results, error = await search_in_collection(
                query=query,
                collection_name=self.collection_name,
                device=self.search_device,
                k=self.k,
                client=client
            )
            
            end_perf_counter = time.perf_counter()
            end_time = time.time()
            
            execution_time = end_time - start_time
            perf_time = end_perf_counter - start_perf_counter
            
            if error:
                logger.error(f"Ошибка при поиске по запросу {query_id}: {error}")
                return query_id, query, execution_time, 0, perf_time
            
            result_count = len(results)
            logger.info(f"Запрос {query_id} выполнен за {execution_time:.3f} секунд, найдено {result_count} результатов")
            
            return query_id, query, execution_time, result_count, perf_time
            
        except Exception as e:
            end_perf_counter = time.perf_counter()
            end_time = time.time()
            
            execution_time = end_time - start_time
            perf_time = end_perf_counter - start_perf_counter
            
            logger.error(f"Ошибка при выполнении запроса {query_id}: {e}")
            return query_id, query, execution_time, 0, perf_time
    
    async def run_sequential_test(self, queries: List[str]) -> List[Tuple[int, str, float, int, float]]:
        """Выполняет последовательный тест."""
        logger.info("Начало последовательного теста...")
        start_total_time = time.time()
        
        results = []
        for i, query in enumerate(queries, 1):
            logger.info(f"Выполнение запроса {i}/{len(queries)}")
            result = await self.single_search(query, i)
            results.append(result)
        
        total_time = time.time() - start_total_time
        logger.info(f"Последовательный тест завершен за {total_time:.3f} секунд")
        
        return results
    
    async def run_concurrent_test(self, queries: List[str], concurrency_level: int = 10) -> List[Tuple[int, str, float, int, float]]:
        """Выполняет параллельный тест с заданным уровнем конкурентности."""
        logger.info(f"Начало параллельного теста с уровнем конкурентности {concurrency_level}...")
        start_total_time = time.time()
        
        # Разбиваем запросы на группы по уровню конкурентности
        results = []
        for i in range(0, len(queries), concurrency_level):
            batch = queries[i:i+concurrency_level]
            batch_indices = list(range(i+1, min(i+concurrency_level+1, len(queries)+1)))
            
            logger.info(f"Выполнение пакета запросов {i+1}-{min(i+concurrency_level, len(queries))}")
            
            # Создаем задачи для текущего пакета
            tasks = [
                self.single_search(query, idx) 
                for query, idx in zip(batch, batch_indices)
            ]
            
            # Выполняем задачи параллельно
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Обрабатываем результаты
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Ошибка при выполнении запроса {batch_indices[j]}: {result}")
                    results.append((batch_indices[j], batch[j], 0, 0, 0))
                else:
                    results.append(result)
        
        total_time = time.time() - start_total_time
        logger.info(f"Параллельный тест завершен за {total_time:.3f} секунд")
        
        return results
    
    def print_results_summary(self, sequential_results: List[Tuple], concurrent_results: List[Tuple]):
        """Выводит сводку результатов."""
        logger.info("=" * 80)
        logger.info("СВОДКА РЕЗУЛЬТАТОВ ТЕСТИРОВАНИЯ")
        logger.info("=" * 80)
        
        # Извлекаем времена выполнения
        seq_times = [result[2] for result in sequential_results]
        conc_times = [result[2] for result in concurrent_results]
        seq_perf_times = [result[4] for result in sequential_results]
        conc_perf_times = [result[4] for result in concurrent_results]
        
        # Считаем общие времена
        sequential_total_time = sum(seq_times)
        concurrent_total_time = max(conc_times) if conc_times else 0
        
        logger.info(f"Общее время последовательных запросов: {sequential_total_time:.3f} секунд")
        logger.info(f"Общее время параллельных запросов: {concurrent_total_time:.3f} секунд")
        
        # Рассчитываем ускорение
        if concurrent_total_time > 0:
            speedup = sequential_total_time / concurrent_total_time
            logger.info(f"Ускорение: {speedup:.2f}x")
        
        # Статистика по времени выполнения
        if seq_times:
            logger.info(f"\nСтатистика последовательных запросов:")
            logger.info(f"  Среднее время: {statistics.mean(seq_times):.3f} секунд")
            logger.info(f"  Медиана: {statistics.median(seq_times):.3f} секунд")
            logger.info(f"  Минимум: {min(seq_times):.3f} секунд")
            logger.info(f"  Максимум: {max(seq_times):.3f} секунд")
        
        if conc_times:
            logger.info(f"\nСтатистика параллельных запросов:")
            logger.info(f"  Среднее время: {statistics.mean(conc_times):.3f} секунд")
            logger.info(f"  Медиана: {statistics.median(conc_times):.3f} секунд")
            logger.info(f"  Минимум: {min(conc_times):.3f} секунд")
            logger.info(f"  Максимум: {max(conc_times):.3f} секунд")
        
        # Статистика по времени выполнения (perf_counter)
        if seq_perf_times:
            logger.info(f"\nСтатистика (perf_counter) последовательных запросов:")
            logger.info(f"  Среднее время: {statistics.mean(seq_perf_times):.3f} секунд")
            logger.info(f"  Медиана: {statistics.median(seq_perf_times):.3f} секунд")
        
        if conc_perf_times:
            logger.info(f"\nСтатистика (perf_counter) параллельных запросов:")
            logger.info(f"  Среднее время: {statistics.mean(conc_perf_times):.3f} секунд")
            logger.info(f"  Медиана: {statistics.median(conc_perf_times):.3f} секунд")
        
        # Рассчитываем пропускную способность
        if sequential_total_time > 0:
            seq_throughput = len(sequential_results) / sequential_total_time
            logger.info(f"\nПропускная способность последовательных запросов: {seq_throughput:.2f} запросов/сек")
        
        if concurrent_total_time > 0:
            conc_throughput = len(concurrent_results) / concurrent_total_time
            logger.info(f"Пропускная способность параллельных запросов: {conc_throughput:.2f} запросов/сек")
            
            if seq_throughput > 0:
                throughput_improvement = conc_throughput / seq_throughput
                logger.info(f"Улучшение пропускной способности: {throughput_improvement:.2f}x")

async def main():
    """Основная функция для запуска тестов."""
    logger.info("Запуск детального теста асинхронности поисковых запросов")
    
    tester = AsyncSearchTester()
    
    logger.info(f"Параметры поиска: коллекция='{tester.collection_name}', устройство='{tester.search_device}', количество результатов={tester.k}")
    
    # Выполняем последовательные запросы
    logger.info("=" * 50)
    logger.info("ТЕСТ 1: Последовательное выполнение запросов")
    logger.info("=" * 50)
    sequential_results = await tester.run_sequential_test(TEST_QUERIES)
    
    # Выполняем параллельные запросы
    logger.info("=" * 50)
    logger.info("ТЕСТ 2: Параллельное выполнение запросов")
    logger.info("=" * 50)
    concurrent_results = await tester.run_concurrent_test(TEST_QUERIES, concurrency_level=5)
    
    # Выводим сводку результатов
    tester.print_results_summary(sequential_results, concurrent_results)
    
    logger.info("Тестирование завершено")

if __name__ == "__main__":
    asyncio.run(main())