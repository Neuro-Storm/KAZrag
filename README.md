# KAZrag

KAZrag — это поисковая система на основе векторного поиска с использованием Qdrant и FastAPI. Проект позволяет обрабатывать PDF-файлы, конвертировать их в Markdown, индексировать текстовые данные и выполнять семантический поиск по коллекциям.

---

## 📦 Структура проекта

- `main.py` — основной скрипт (FastAPI + логика индексации и поиска)
- `core/` — основные модули приложения
  - `embeddings.py` — работа с эмбеддерами
  - `gguf_embeddings.py` — работа с GGUF моделями
  - `indexer.py` — индексация документов
  - `searcher.py` — поиск по коллекциям
  - `chunker.py` — нарезка текста на чанки
  - `pdf_to_md_chunker.py` — обработка PDF и конвертация в Markdown
  - `multi_format_converter.py` — конвертация файлов различных форматов в Markdown
  - `file_converter.py` — конвертация PDF с использованием настроек
  - `qdrant_collections.py` — управление коллекциями Qdrant
  - `dependencies.py` — зависимости FastAPI
  - `converters/` — модули для конвертации различных форматов
    - `base.py` — базовый класс для конвертеров
    - `docx_converter.py` — конвертер DOCX в Markdown
    - `txt_converter.py` — конвертер TXT в Markdown
    - `html_converter.py` — конвертер HTML в Markdown
    - `djvu_converter.py` — конвертер DJVU в PDF
    - `image_converter.py` — обработчик изображений (JPG, PNG)
    - `presentation_converter.py` — конвертер презентаций (PPT, PPTX) в Markdown
    - `excel_converter.py` — конвертер таблиц (XLSX, XLS) в Markdown
    - `manager.py` — менеджер конвертеров
- `web/` — веб-приложения FastAPI
  - `search_app.py` — приложение для поиска
  - `admin_app.py` — приложение для администрирования
- `config/` — конфигурация приложения
  - `settings.py` — работа с настройками (Pydantic модель)
- `templates/` — HTML-шаблоны для веб-интерфейса
  - `index.html` — страница поиска
  - `settings.html` — страница настроек
- `requirements.txt` — зависимости Python
- `config.json` — конфигурация приложения
- `.env` — переменные окружения
- `.env.example` — пример файла переменных окружения
- `.gitignore` — исключения для Git
- `LICENSE` — лицензия MIT

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
venv\Scripts\activate
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

### 4. Настройка моделей

### 5. Установка и запуск Qdrant

Запустите Qdrant локально с помощью Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

Или используйте облачную версию Qdrant, указав URL в `config.json`.

### 4. Установка MinerU (для обработки PDF)

```bash
pip install uv
uv pip install -U "mineru[core]"
```

> **Важно:** MinerU требует Python < 3.13 и может устанавливать дополнительные зависимости (`magic-pdf`).

### 5. Настройка переменных окружения

Скопируйте файл `.env.example` в `.env` и при необходимости измените настройки:

```bash
cp .env.example .env
```

В файле `.env` можно настроить:
- `ALLOWED_ORIGINS` — разрешенные источники для CORS
- `ADMIN_API_KEY` — API ключ для доступа к админке

---

## 🛠️ Использование

1. Убедитесь, что Qdrant запущен.
2. Активируйте виртуальное окружение (`venv\Scripts\activate`).
3. Поместите файлы различных форматов для обработки в папку, указанную в настройках (`pdfs_to_process/` по умолчанию). Поддерживаются следующие форматы:
   - PDF (обрабатываются напрямую MinerU)
   - DJVU (конвертируются в PDF, затем обрабатываются MinerU)
   - Изображения JPG, PNG (обрабатываются напрямую MinerU)
   - Документы DOCX, TXT, HTML (конвертируются в Markdown)
   - Презентации PPT, PPTX (конвертируются в Markdown)
   - Таблицы XLSX, XLS (конвертируются в Markdown)
4. Запустите FastAPI-приложение:

```bash
python main.py
```

- Веб-интерфейс будет доступен по адресу: [http://localhost:8000](http://localhost:8000)
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

Для доступа к админке используйте API ключ, указанный в файле `.env`.

---

## ⚙️ Основные возможности

- Обработка и разбиение PDF на Markdown (MinerU)
- Обработка файлов различных форматов:
  - Документы: DOCX, TXT, HTML
  - Изображения: JPG, PNG (с помощью MinerU)
  - Презентации: PPT, PPTX
  - Таблицы: XLSX, XLS
  - DJVU (конвертируется в PDF, затем в Markdown)
- Индексация текстовых и Markdown файлов в Qdrant
- Семантический поиск по коллекциям
- Веб-интерфейс для поиска и управления настройками
- Гибкая настройка моделей, параметров индексации и путей
- Поддержка GGUF моделей для эмбеддингов
- Безопасность через API ключи
- Логирование и обработка ошибок

---



- [Qdrant — документация](https://qdrant.tech/)
- [MinerU — GitHub](https://github.com/opendatalab/MinerU)

---

## 📝 Лицензия

Проект распространяется под лицензией MIT. См. файл [LICENSE](LICENSE).
