"""Модуль для централизованного управления конфигурацией с использованием pydantic-settings."""

import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from cachetools import TTLCache
from pydantic import ValidationError

from config.settings import Config
from config.resource_path import resource_path

logger = logging.getLogger(__name__)


class ConfigManager:
    """Централизованный менеджер конфигурации с кэшированием и валидацией с использованием pydantic-settings."""
    
    _instance: Optional['ConfigManager'] = None
    
    def __init__(self, cache_ttl: int = 60):
        """Инициализировать ConfigManager.
        
        Args:
            cache_ttl: Время жизни кэшированной конфигурации в секундах
        """
        self.cache_ttl = cache_ttl
        # Использовать TTLCache для кэширования конфигурации
        self._cache = TTLCache(maxsize=1, ttl=cache_ttl)
        self._config_instance: Optional[Config] = None
        
    @classmethod
    def get_instance(cls, cache_ttl: int = 60) -> 'ConfigManager':
        """Получить singleton экземпляр ConfigManager.
        
        Args:
            cache_ttl: Время жизни кэшированной конфигурации в секундах
            
        Returns:
            ConfigManager: Singleton экземпляр
        """
        if cls._instance is None:
            cls._instance = cls(cache_ttl)
        return cls._instance
    
    def _load_config_from_file(self) -> Config:
        """Загрузить конфигурацию из файла без кэширования.
        
        Returns:
            Config: Загруженный объект конфигурации
        """
        logger.debug("Загрузка конфигурации из файла")
        config_file = resource_path("config/config.json")
        
        if not config_file.exists():
            # Создать конфигурацию по умолчанию, если файл не существует
            config = Config()
            self.save(config)
            return config
            
        try:
            with open(config_file, encoding="utf-8") as f:
                config_dict = json.load(f)
                
            # Создать объект Config с валидацией
            config = Config(**config_dict)
            return config
            
        except ValidationError as e:
            logger.exception(f"Ошибки валидации конфигурации: {e.errors()}")
            # Вернуть конфигурацию по умолчанию, если валидация не удалась
            config = Config()
            self.save(config)
            return config
            
        except Exception as e:
            logger.exception(f"Ошибка загрузки конфигурации: {e}")
            # Вернуть конфигурацию по умолчанию, если произошла любая другая ошибка
            config = Config()
            self.save(config)
            return config
    
    def _load_config_from_settings(self) -> Config:
        """Загрузить конфигурацию из переменных окружения и настроек.
        
        Returns:
            Config: Загруженный объект конфигурации
        """
        try:
            # Загрузить конфигурацию из переменных окружения и значений по умолчанию
            config = Config()
            return config
        except Exception as e:
            logger.exception(f"Ошибка загрузки конфигурации из настроек: {e}")
            # Откат к конфигурации на основе файла
            return self._load_config_from_file()
    
    def load(self) -> Config:
        """Загрузить конфигурацию из файла с кэшированием.
        
        Returns:
            Config: Загруженный объект конфигурации
        """
        # Проверить, есть ли у нас действительная кэшированная конфигурация
        if 'config' in self._cache:
            logger.debug("Возврат кэшированной конфигурации")
            return self._cache['config']
        
        # Сначала попробовать загрузить из файла
        try:
            config = self._load_config_from_file()
        except Exception as e:
            logger.warning(f"Не удалось загрузить конфигурацию из файла, возврат к настройкам: {e}")
            # Откат к конфигурации на основе настроек
            config = self._load_config_from_settings()
        
        # Обновить кэш
        self._cache['config'] = config
        
        return config
    
    def save(self, config: Config) -> None:
        """Сохранить конфигурацию в файл.
        
        Args:
            config: Объект конфигурации для сохранения
        """
        try:
            # Сохранить в файл
            config.save_to_file()
            
            # Обновить кэш
            self._cache['config'] = config
            
            logger.info("Конфигурация успешно сохранена")
            
        except Exception as e:
            logger.exception(f"Ошибка сохранения конфигурации: {e}")
            raise
    
    def get(self) -> Config:
        """Получить конфигурацию (алиас для load).
        
        Returns:
            Config: Объект конфигурации
        """
        return self.load()
    
    def reload(self) -> Config:
        """Принудительно перезагрузить конфигурацию из файла, обходя кэш.
        
        Returns:
            Config: Перезагруженный объект конфигурации
        """
        # Очистить кэш
        self._cache.clear()
        
        # Загрузить свежую конфигурацию
        return self.load()
    
    def update_from_dict(self, updates: Dict[str, Any]) -> Config:
        """Обновить конфигурацию из словаря.
        
        Args:
            updates: Словарь с обновлениями конфигурации
            
        Returns:
            Config: Обновленный объект конфигурации
        """
        try:
            # Получить текущую конфигурацию
            current_config = self.get()
            
            # Преобразовать в словарь и обновить
            config_dict = current_config.model_dump()
            config_dict.update(updates)
            
            # Создать новый объект конфигурации
            updated_config = Config(**config_dict)
            
            # Сохранить обновленную конфигурацию
            self.save(updated_config)
            
            return updated_config
            
        except Exception as e:
            logger.exception(f"Ошибка обновления конфигурации: {e}")
            raise