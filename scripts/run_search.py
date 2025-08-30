#!/usr/bin/env python

"""
Скрипт для выполнения поиска в коллекции Qdrant
"""

import logging
import os
import sys

# Добавляем корневую директорию проекта в sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import load_config
from core.search.searcher import search_in_collection

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def run_search():
    """Основная функция для выполнения поиска"""
    try:
        logger.info("Начало процесса поиска")
        
        # Загружаем конфигурацию
        load_config()
        logger.info("Конфигурация загружена успешно")
        
        # Устанавливаем параметры поиска
        collection_name = "collection_384"  # Используем указанную коллекцию
        search_query = "проверка"
        device = "cuda"  # Используем GPU
        k = 5  # Количество результатов
        hybrid = True  # Гибридный режим поиска
        
        logger.info(f"Параметры поиска: коллекция={collection_name}, запрос='{search_query}', "
                   f"устройство={device}, результатов={k}, гибридный_режим={hybrid}")
        
        # Выполняем поиск
        results, error = await search_in_collection(
            query=search_query,
            collection_name=collection_name,
            device=device,
            k=k,
            hybrid=hybrid
        )
        
        if error:
            logger.error(f"Ошибка при поиске: {error}")
            return 1
            
        # Выводим результаты
        logger.info(f"Поиск завершен успешно. Найдено результатов: {len(results)}")
        
        if results:
            print("\n" + "="*80)
            print(f"РЕЗУЛЬТАТЫ ПОИСКА ПО ЗАПРОСУ: '{search_query}'")
            print("="*80)
            
            for i, (result, score) in enumerate(results, 1):
                content = result.get('content', result.get('page_content', ''))
                source = result.get('metadata', {}).get('source', 'N/A')
                
                print(f"\n{i}. ИСТОЧНИК: {source}")
                print(f"   СХОДСТВО: {score:.4f}")
                print(f"   СОДЕРЖИМОЕ: {content[:200]}{'...' if len(content) > 200 else ''}")
                
                # Если есть метаданные, выводим их
                metadata = result.get('metadata', {})
                if metadata:
                    print("   МЕТАДАННЫЕ:")
                    for key, value in metadata.items():
                        if key != 'source':  # Источник уже вывели отдельно
                            print(f"     {key}: {value}")
        else:
            print(f"\nПо запросу '{search_query}' ничего не найдено.")
            
        return 0  # Успешное завершение
        
    except Exception as e:
        logger.error(f"Ошибка при выполнении поиска: {e}")
        return 1  # Завершение с ошибкой


def main():
    """Точка входа в скрипт"""
    import asyncio
    exit_code = asyncio.run(run_search())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()