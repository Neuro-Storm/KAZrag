# KAZrag

KAZrag — это поисковая система на основе векторного поиска с использованием Qdrant и FastAPI. Проект позволяет обрабатывать PDF-файлы, конвертировать их в Markdown, индексировать текстовые данные и выполнять семантический поиск по коллекциям.

---

## 📦 Структура проекта

- `main.py` — основной скрипт (FastAPI + логика индексации и поиска)
- `app/` — модуль веб-приложения
  - `app_factory.py` — создание и конфигурация FastAPI приложения
  - `routes.py` — регистрация маршрутов
  - `startup.py` — обработчики событий запуска и остановки
- `core/` — основные модули приложения
  - `embedding/` — работа с эмбеддерами
    - `embeddings.py` — инициализация эмбеддеров
    - `gguf_embeddings.py` — работа с GGUF моделями
    - `sparse_embedding_adapter.py` — адаптер для разреженных эмбеддингов
  - `indexing/` — индексация документов
    - `indexer.py` — основной индексатор
    - `indexer_component.py` — компоненты индексации
    - `multilevel_indexer.py` — многоуровневый индексатор
    - `document_loader.py` — загрузчик документов
    - `metadata_manager.py` — менеджер метаданных
    - `text_splitter.py` — разделение текста на чанки
    - `chunker.py` — базовый чанкер
    - `paragraph_chunker.py` — чанкер по абзацам
    - `sentence_chunker.py` — чанкер по предложениям
    - `multilevel_chunker.py` — многоуровневый чанкер
    - `indexer_additional.py` — дополнительные функции индексации
  - `search/` — поиск по коллекциям
    - `searcher.py` — основной поиск
    - `search_strategy.py` — стратегии поиска
    - `search_executor.py` — исполнитель поиска
    - `collection_analyzer.py` — анализ коллекций
  - `converting/` — конвертация файлов различных форматов
    - `multi_format_converter.py` — конвертация файлов различных форматов в Markdown
    - `file_converter.py` — базовый конвертер файлов
    - `file_processor.py` — процессор файлов
    - `manager.py` — менеджер конвертеров
    - `mineru_service.py` — сервис для работы с MinerU
    - `pdf_to_md_chunker.py` — чанкер PDF в Markdown
    - `converters/` — модули для конвертации различных форматов
      - `base.py` — базовый класс для конвертеров
      - `djvu_converter.py` — конвертер DJVU в PDF
      - `unstructured_converter.py` — универсальный конвертер
  - `qdrant/` — взаимодействие с Qdrant
    - `qdrant_client.py` — клиент Qdrant
    - `qdrant_collections.py` — управление коллекциями Qdrant
    - `qdrant_collections_additional.py` — дополнительные функции управления коллекциями
  - `utils/` — вспомогательные модули
    - `constants.py` — константы приложения
    - `exception_handlers.py` — обработчики исключений
    - `collection_manager.py` — менеджер коллекций
    - `dependencies.py` — зависимости FastAPI
- `web/` — веб-приложения FastAPI
  - `search_app.py` — приложение для поиска
  - `admin_app.py` — приложение для администрирования
- `scripts/` — вспомогательные скрипты
  - `run_search.py` — скрипт для выполнения поиска
  - `run_indexing.py` — скрипт для запуска индексации
  - `clear_log.py` — скрипт для очистки логов
- `config/` — конфигурация приложения
  - `settings.py` — работа с настройками (Pydantic модель)
  - `config_manager.py` — централизованный менеджер конфигурации
  - `logging_config.py` — конфигурация логирования
- `templates/` — HTML-шаблоны для веб-интерфейса
  - `index.html` — страница поиска
  - `search_results.html` — страница результатов поиска
  - `settings.html` — страница настроек
- `requirements.txt` — зависимости Python
- `pyproject.toml` — конфигурация проекта
- `.env.example` — пример файла переменных окружения
- `.gitignore` — исключения для Git
- `LICENSE` — лицензия MIT

---

## 🚀 Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/Neuro-Storm/KAZrag.git
cd KAZrag
```

### 2. Установка зависимостей

Рекомендуется использовать виртуальное окружение:

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# или
source venv/bin/activate  # Linux/Mac
```

Установите зависимости:

```bash
pip install -r requirements.txt
```

### 3. Настройка переменных окружения

Скопируйте `.env.example` в `.env` и при необходимости отредактируйте:

```bash
cp .env.example .env
```

В `.env` можно настроить:
- `ADMIN_API_KEY` для защиты админки
- `FASTEMBED_CACHE_DIR` для указания директории кэша моделей BM25 (по умолчанию `./models/fastembed_cache`)

### 4. Установка и запуск Qdrant

Запустите Qdrant локально с помощью Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

Или используйте облачную версию Qdrant, указав URL в конфигурации.

### 5. Установка MinerU (для обработки PDF)

```bash
pip install uv
uv pip install -U "mineru[core]"
```

> **Важно:** MinerU требует Python < 3.13 и может устанавливать дополнительные зависимости (`magic-pdf`).

---

## 🛠️ Использование

1. Убедитесь, что Qdrant запущен.
2. Активируйте виртуальное окружение (`venv\Scripts\activate` или `source venv/bin/activate`).
3. Поместите файлы различных форматов для обработки в папку `pdfs_to_process/`. Поддерживаются следующие форматы:
   - PDF (обрабатываются напрямую MinerU)
   - DJVU (конвертируются в PDF, затем обрабатываются MinerU)
   - Изображения JPG, PNG (обрабатываются напрямую MinerU)
   - Документы DOCX, TXT, HTML (конвертируются в Markdown)
   - Презентации PPT, PPTX (конвертируются в Markdown)
   - Таблицы XLSX, XLS (конвертируются в Markdown)

### Веб-интерфейс

Запустите FastAPI-приложение:

```bash
python main.py
```

- Веб-интерфейс будет доступен по адресу: [http://localhost:8000](http://localhost:8000)
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Админка: [http://localhost:8000/settings](http://localhost:8000/settings)

В веб-интерфейсе доступны следующие возможности:
- Поиск по коллекциям с фильтрацией по метаданным
- Управление настройками индексации, включая выбор модели для плотных векторов
- Управление настройками конвертации файлов различных форматов
- Запуск процесса индексации и обработки файлов
- Просмотр информации о коллекциях

### Консольный интерфейс

Альтернативно, вы можете использовать консольный интерфейс для поиска:

```bash
python consolemain.py
```

Консольный интерфейс предоставляет те же функции поиска, что и веб-интерфейс, но в текстовом режиме.

Для доступа к админке используйте API ключ, указанный в файле `.env`.

---

## 🧪 Тестирование

Проект включает в себя как unit-тесты, так и интеграционные тесты для обеспечения качества кода.

### Запуск тестов

Для запуска всех тестов используйте pytest:

```bash
# Запуск всех тестов
python -m pytest tests -v

# Запуск тестов с отчетом о покрытии
python -m pytest tests --cov=core --cov=web --cov=config --cov-report=html

# Запуск только unit-тестов
python -m pytest tests/test_config_manager.py tests/test_collection_manager.py tests/test_embedding_manager.py tests/test_indexer_components.py tests/test_search_components.py tests/test_web_routes.py tests/test_paragraph_chunker.py tests/test_sentence_chunker.py -v

# Запуск только интеграционных тестов
python -m pytest tests/test_indexing_integration.py tests/test_search_integration.py tests/test_web_api_integration.py -v
```

---

## 🔄 CI/CD и линтинг

Проект использует GitHub Actions для непрерывной интеграции и непрерывной доставки.

### Линтинг кода

Проект использует `ruff` для линтинга кода. Конфигурация находится в файле `ruff.toml`.

Для запуска линтера:

```bash
# Установка ruff
pip install ruff

# Запуск линтера
ruff check .

# Автоматическое исправление ошибок (где возможно)
ruff check . --fix
```

---

## ⚙️ Основные возможности

- Обработка и разбиение PDF на Markdown (MinerU)
- Обработка файлов различных форматов:
  - Документы: DOCX, TXT, HTML
  - Изображения: JPG, PNG (с помощью MinerU)
  - Презентации: PPT, PPTX
  - Таблицы: XLSX, XLS
  - DJVU (конвертируется в PDF, затем в Markdown)
- Индексация текстовых и Markdown файлов в Qdrant с извлечением метаданных
- Семантический поиск по коллекциям с возможностью фильтрации по метаданным
- Веб-интерфейс для поиска и управления настройками
- Гибкая настройка моделей, параметров индексации и путей
- Поддержка GGUF моделей для эмбеддингов
- Безопасность через API ключи
- Логирование и обработка ошибок
- Централизованное управление конфигурацией с кэшированием
- Различные стратегии чанкинга:
  - По символам (стандартная)
  - По абзацам (с настраиваемым количеством абзацев в чанке и перекрытием)
  - По предложениям (с настраиваемым количеством предложений в чанке и перекрытием)
- Многоуровневый чанкинг (один фрагмент - несколько векторов):
  - Полностью настраиваемые макро-чанки (размер, стратегия, перекрытие)
  - Полностью настраиваемые микро-чанки (размер, стратегия, перекрытие)
  - Любая комбинация стратегий на разных уровнях
  - Интуитивный веб-интерфейс с разделением настроек
- Настройка параметров чанкинга через веб-интерфейс
- История использованных моделей для быстрого выбора
- Отслеживание состояния индексации

---

## 📚 Полезные ссылки

- [Qdrant — документация](https://qdrant.tech/)
- [MinerU — GitHub](https://github.com/opendatalab/MinerU)

---

## 📝 Лицензия

Проект распространяется под лицензией MIT. См. файл [LICENSE](LICENSE).