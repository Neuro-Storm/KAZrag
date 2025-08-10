"""Модуль для работы с конфигурацией приложения."""

import json
from pathlib import Path
from typing import Dict, Any

CONFIG_FILE = Path("config.json")


def load_config() -> Dict[str, Any]:
    """Загружает конфигурацию из файла config.json.
    
    Если файл не существует, создает его с настройками по умолчанию.
    """
    if not CONFIG_FILE.exists():
        default_cfg = {
            "folder_path": "./data_to_index",
            "collection_name": "final-dense-collection",
            "current_hf_model": "sentence-transformers/all-MiniLM-L6-v2",
            "hf_model_history": ["sentence-transformers/all-MiniLM-L6-v2"],
            "chunk_size": 500,
            "chunk_overlap": 100,
            "qdrant_url": "http://localhost:6333",
            "device": "auto",
            "use_dense_vectors": True,
            "is_indexed": False,
            "mineru_input_pdf_dir": "./pdfs_to_process",
            "mineru_output_md_dir": "./data_to_index",
            "mineru_enable_formula_parsing": False,
            "mineru_enable_table_parsing": False,
            "mineru_model_source": "huggingface",
            "mineru_models_dir": "",
            "mineru_backend": "pipeline",
            "mineru_method": "auto",
            "mineru_lang": "east_slavic",
            "mineru_sglang_url": "",
            "embedding_batch_size": 32,
            "indexing_batch_size": 50,
        }
        save_config(default_cfg)
        return default_cfg
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: Dict[str, Any]):
    """Сохраняет конфигурацию в файл config.json."""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)