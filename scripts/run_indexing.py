#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Скрипт для запуска индексации документов"""

import sys
import os
import logging
from pathlib import Path

# Добавляем корневую директорию проекта в sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import load_config
from core.indexing.indexer import run_indexing_from_config
from core.utils.constants import DEFAULT_COLLECTION_NAME

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Основная функция для запуска индексации"""
    try:
        logger.info("Начало процесса индексации")
        
        # Загружаем конфигурацию
        config = load_config()
        logger.info("Конфигурация загружена успешно")
        
        # Проверяем путь к папке с документами
        folder_path = Path(config.folder_path)
        logger.info(f"Путь к папке с документами: {folder_path}")
        
        if not folder_path.exists():
            logger.error(f"Папка с документами не существует: {folder_path}")
            return 1
            
        if not folder_path.is_dir():
            logger.error(f"Указанный путь не является папкой: {folder_path}")
            return 1
            
        # Подсчитываем количество файлов
        txt_files = list(folder_path.rglob("*.txt"))
        md_files = list(folder_path.rglob("*.md"))
        logger.info(f"Найдено .txt файлов: {len(txt_files)}")
        logger.info(f"Найдено .md файлов: {len(md_files)}")
        
        # Показываем первые 5 файлов каждого типа
        if txt_files:
            logger.info("Первые 5 .txt файлов:")
            for i, file in enumerate(txt_files[:5]):
                logger.info(f"  {i+1}. {file}")
                
        if md_files:
            logger.info("Первые 5 .md файлов:")
            for i, file in enumerate(md_files[:5]):
                logger.info(f"  {i+1}. {file}")
        
        # Запускаем индексацию
        success, status = run_indexing_from_config()
        
        if success:
            logger.info("Индексация завершена успешно")
            logger.info(f"Статус: {status}")
            return 0  # Успешное завершение
        else:
            logger.error(f"Индексация завершена с ошибкой: {status}")
            return 1  # Завершение с ошибкой
        
    except Exception as e:
        logger.error(f"Ошибка при индексации: {e}")
        return 1  # Завершение с ошибкой

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)