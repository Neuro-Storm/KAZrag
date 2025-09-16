"""Модуль для работы с конфигурацией приложения - совместимость с предыдущей версией."""

from config.settings_model import Config
from config.config_manager import ConfigManager

# Для обратной совместимости
CONFIG_FILE = "config/config.json"

# Функция для обратной совместимости
def load_config(reload: bool = False) -> Config:
    """Загрузить конфигурацию для обратной совместимости.
    
    Args:
        reload: Если True, принудительно перезагрузить конфигурацию
        
    Returns:
        Config: Объект конфигурации
    """
    manager = ConfigManager.get_instance()
    if reload:
        return manager.reload()
    return manager.get()

__all__ = ["Config", "ConfigManager", "CONFIG_FILE", "load_config"]