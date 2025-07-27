# KAZrag

Этот проект предназначен для обработки PDF-файлов, их конвертации в markdown и дальнейшей работы с полученными данными.

## Структура проекта

- `main.py` — основной скрипт запуска
- `pdf_to_md_chunker.py` — модуль для конвертации PDF в markdown
- `requirements.txt` — зависимости проекта
- `config.json` — конфигурационный файл
- `data_to_index/` — директория с обработанными файлами
- `pdfs_to_process/` — директория для исходных PDF
- `templates/` — HTML-шаблоны

## Быстрый старт

1. Клонируйте репозиторий:
   ```sh
   git clone https://github.com/yourusername/KAZrag.git
   cd KAZrag
   ```
2. Установите зависимости:
   ```sh
   pip install -r requirements.txt
   ```
3. Поместите PDF-файлы в папку `pdfs_to_process/`.
4. Запустите основной скрипт:
   ```sh
   python main.py
   ```

## Лицензия

MIT License
