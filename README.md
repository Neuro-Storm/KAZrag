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
  - `file_converter.py` — конвертация PDF с использованием настроек
  - `qdrant_collections.py` — управление коллекциями Qdrant
  - `dependencies.py` — зависимости FastAPI
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
pip install -r requirements.txt
```

### 3. Установка и запуск Qdrant

- Скачайте [Qdrant для Windows](https://github.com/qdrant/qdrant/releases) (файл `qdrant-x86_64-pc-windows-msvc.tar.gz`).
- Распакуйте и запустите `qdrant.exe` (порт 6333 по умолчанию).

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
3. Поместите PDF-файлы для обработки в папку, указанную в настройках (`pdfs_to_process/` по умолчанию).
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
- Индексация текстовых и Markdown файлов в Qdrant
- Семантический поиск по коллекциям
- Веб-интерфейс для поиска и управления настройками
- Гибкая настройка моделей, параметров индексации и путей
- Поддержка GGUF моделей для эмбеддингов
- Безопасность через API ключи
- Логирование и обработка ошибок

---

## 📚 Полезные ссылки

- [Qdrant — документация](https://qdrant.tech/)
- [MinerU — GitHub](https://github.com/opendatalab/MinerU)

---

## 📝 Лицензия

Проект распространяется под лицензией MIT. См. файл [LICENSE](LICENSE).
