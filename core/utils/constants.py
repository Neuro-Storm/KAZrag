"""Модуль с константами проекта."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

# Порог памяти для определения размера батча (500MB по умолчанию)
MEMORY_THRESHOLD = 500 * 1024 * 1024

# Количество результатов поиска по умолчанию
DEFAULT_K = 5

# Размер батча для обработки документов по умолчанию
DEFAULT_BATCH_SIZE = 32

# URL Qdrant по умолчанию
QDRANT_DEFAULT_URL = "http://localhost:6333"

# Имя коллекции по умолчанию
DEFAULT_COLLECTION_NAME = "final-dense-collection"

# Количество попыток повтора для подключения к Qdrant
RETRY_ATTEMPTS = 3

# Время ожидания между попытками повтора (в секундах)
RETRY_WAIT_TIME = 2

# Настройка Jinja2 шаблонизатора
TEMPLATES_DIR = Path(__file__).parent.parent / "web" / "templates"
TEMPLATES = Environment(loader=FileSystemLoader(TEMPLATES_DIR))