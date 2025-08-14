"""Модуль для работы с конфигурацией приложения."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_FILE = Path("config.json")


class Config(BaseModel):
    """Модель конфигурации приложения."""
    
    # Настройки индексации
    folder_path: str = "./data_to_index"
    collection_name: str = "final-dense-collection"
    current_hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_model_history: List[str] = Field(default_factory=lambda: ["sentence-transformers/all-MiniLM-L6-v2"])
    chunk_size: int = 500
    chunk_overlap: int = 100
    device: str = "auto"
    use_dense_vectors: bool = True
    is_indexed: bool = False
    embedding_batch_size: int = 32
    indexing_batch_size: int = 50
    force_recreate: bool = True
    
    # Настройки Qdrant
    qdrant_url: str = "http://localhost:6333"
    
    # Настройки MinerU
    mineru_input_pdf_dir: str = "./pdfs_to_process"
    mineru_output_md_dir: str = "./data_to_index"
    mineru_enable_formula_parsing: bool = False
    mineru_enable_table_parsing: bool = False
    mineru_model_source: str = "huggingface"
    mineru_models_dir: str = ""
    mineru_backend: str = "pipeline"
    mineru_method: str = "auto"
    mineru_lang: str = "east_slavic"
    mineru_sglang_url: str = ""


def load_config() -> Config:
    """Загружает конфигурацию из файла config.json.
    
    Если файл не существует, создает его с настройками по умолчанию.
    """
    if not CONFIG_FILE.exists():
        default_cfg = Config()
        save_config(default_cfg)
        return default_cfg
    
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
        
    # Создаем объект Config из словаря
    try:
        config = Config(**config_dict)
    except Exception as e:
        logger.error(f"Ошибка валидации конфигурации: {e}")
        # Создаем конфигурацию по умолчанию и сохраняем её
        config = Config()
        save_config(config)
        return config
        
    # Проверяем доступность Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=config.qdrant_url)
        # Попытка подключения к Qdrant
        client.get_collections()
    except Exception as e:
        raise RuntimeError(f"Qdrant не доступен по адресу {config.qdrant_url}. Проверьте, запущен ли сервис Qdrant. Ошибка: {e}")
        
    return config


def save_config(config: Config):
    """Сохраняет конфигурацию в файл config.json."""
    # Преобразуем Config в словарь
    config_dict = config.model_dump()
    
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)