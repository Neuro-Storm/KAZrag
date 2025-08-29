
"""
Модуль для преобразования PDF в Markdown с помощью утилиты MinerU.

Основные функции:
- Конвертирует PDF файлы в Markdown формат
- Сохраняет изображения в отдельную папку
- Поддерживает различные настройки парсинга (формулы, таблицы)
- Обрабатывает файлы через subprocess вызов mineru
- Извлекает метаданные из PDF файлов

Структура выходных данных:
output_md_dir/
  ├── filename/
  │   ├── filename.md       # Основной markdown файл
  │   └── images/           # Папка с извлеченными изображениями
"""

import os
import subprocess
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

logger = logging.getLogger(__name__)

def setup_env(model_source: str, models_dir: Optional[str], enable_formula_parsing: bool, enable_table_parsing: bool) -> dict:
    """
    Настраивает переменные окружения для MinerU.
    
    Args:
        model_source (str): Источник моделей.
        models_dir (Optional[str]): Путь к локальным моделям.
        enable_formula_parsing (bool): Включить парсинг формул.
        enable_table_parsing (bool): Включить парсинг таблиц.
        
    Returns:
        dict: Настроенные переменные окружения.
    """
    env = os.environ.copy()
    env["MINERU_MODEL_SOURCE"] = model_source
    if models_dir and model_source == "local":
        env["MINERU_MODELS_DIR"] = models_dir

    # Настройка флагов парсинга формул и таблиц через переменные окружения
    env["MINERU_ENABLE_FORMULA_PARSING"] = "true" if enable_formula_parsing else "false"
    env["MINERU_ENABLE_TABLE_PARSING"] = "true" if enable_table_parsing else "false"
    
    return env


def run_subprocess(pdf_file: Path, temp_output_dir: Path, backend: str, method: str, lang: str, sglang_url: Optional[str], env: dict) -> subprocess.CompletedProcess:
    """
    Запускает subprocess вызов mineru.
    
    Args:
        pdf_file (Path): Путь к PDF файлу.
        temp_output_dir (Path): Временная директория для вывода.
        backend (str): Бэкенд обработки.
        method (str): Метод парсинга.
        lang (str): Язык для OCR.
        sglang_url (Optional[str]): URL сервера sglang.
        env (dict): Переменные окружения.
        
    Returns:
        subprocess.CompletedProcess: Результат выполнения subprocess.
    """
    logger.info(f"Обработка файла (Subprocess): {pdf_file.name}")

    # Формирование команды для вызова mineru с указанными параметрами
    cmd = [
        "mineru",
        "-p", str(pdf_file),
        "-o", str(temp_output_dir), # Выводим во временную директорию
        "-b", backend,
        "--output-format", "md" # Указываем формат вывода
    ]
    
    if method != "auto":
         cmd.extend(["--method", method])
    
    if lang and lang.lower() not in ['auto', 'none']:
         cmd.extend(["--lang", lang])
         
    if backend == "vlm-sglang-client" and sglang_url:
        cmd.extend(["-u", sglang_url])

    logger.info(f"  -> Выполняется команда: {' '.join(cmd)}")

    return subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=600  # Фиксированный таймаут 10 минут
    )

def postprocess_output(pdf_file: Path, pdf_stem: str, temp_output_dir: Path, final_output_dir: Path, output_root_path: Path) -> bool:
    """
    Пост-обработка вывода mineru.
    
    Args:
        pdf_file (Path): Путь к PDF файлу.
        pdf_stem (str): Имя PDF файла без расширения.
        temp_output_dir (Path): Временная директория с результатами.
        final_output_dir (Path): Финальная директория для результатов.
        output_root_path (Path): Корневая директория для вывода.
        
    Returns:
        bool: Успешность пост-обработки.
    """
    # Пост-обработка: перенос результатов из временной папки в финальную
    # MinerU создает вложенные директории вида temp_output_dir/{pdf_stem}/auto/
    mineru_output_dir = temp_output_dir / pdf_stem / "auto"
    
    # Если такой директории нет, ищем .md файл в temp_output_dir или подпапках
    if not mineru_output_dir.exists():
        md_files = list(temp_output_dir.rglob("*.md"))
        if not md_files:
            logger.warning(f"  -> Предупреждение: Не найден .md файл после обработки {pdf_file.name}.")
            return False
            
        # Используем директорию с первым найденным .md файлом
        mineru_output_dir = md_files[0].parent
        logger.info(f"  -> Используется директория с найденным файлом: {mineru_output_dir.relative_to(temp_output_dir)}")
    else:
        logger.info(f"  -> Используется стандартная директория MinerU: {mineru_output_dir.relative_to(temp_output_dir)}")

    # Проверяем наличие .md файла
    target_md_file = mineru_output_dir / f"{pdf_stem}.md"
    if not target_md_file.exists():
        # Ищем любой .md файл в директории
        md_files = list(mineru_output_dir.glob("*.md"))
        if not md_files:
            logger.warning(f"  -> Предупреждение: Не найден .md файл в директории {mineru_output_dir}.")
            return False
        target_md_file = md_files[0]
        logger.info(f"  -> Используется найденный файл: {target_md_file.name}")

    # Создаем финальную папку для этого PDF
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Копируем .md файл
    final_md_path = final_output_dir / f"{pdf_stem}.md"
    shutil.copy2(target_md_file, final_md_path)  # Используем copy2 для сохранения метаданных
    logger.info(f"  -> .md файл скопирован в: {final_md_path.relative_to(output_root_path)}")
    
    # Копируем папку с изображениями, если она есть
    images_dir_in_temp = mineru_output_dir / "images"
    if images_dir_in_temp.exists() and images_dir_in_temp.is_dir():
        images_dir_final = final_output_dir / "images"
        if images_dir_final.exists():
            shutil.rmtree(images_dir_final)
        shutil.copytree(images_dir_in_temp, images_dir_final)
        logger.info(f"  -> Папка 'images' скопирована в: {images_dir_final.relative_to(output_root_path)}")
        
    return True

def extract_pdf_metadata(pdf_file: Path) -> Dict[str, Any]:
    """
    Извлекает метаданные из PDF файла.
    
    Args:
        pdf_file (Path): Путь к PDF файлу.
        
    Returns:
        Dict[str, Any]: Словарь с метаданными PDF.
    """
    if not HAS_PYPDF2:
        logger.warning("PyPDF2 is not available for metadata extraction")
        return {}
        
    if not pdf_file.exists():
        logger.warning(f"File not found for metadata extraction: {pdf_file}")
        return {}
        
    try:
        metadata = {}
        with open(pdf_file, 'rb') as f:
            pdf = PdfReader(f)
            if pdf.metadata:
                # Извлекаем стандартные метаданные
                pdf_metadata = {
                    "title": pdf.metadata.get("/Title"),
                    "author": pdf.metadata.get("/Author"),
                    "subject": pdf.metadata.get("/Subject"),
                    "creator": pdf.metadata.get("/Creator"),
                    "producer": pdf.metadata.get("/Producer"),
                    "creation_date": pdf.metadata.get("/CreationDate"),
                    "modification_date": pdf.metadata.get("/ModDate"),
                }
                # Фильтруем None значения
                metadata.update({k: v for k, v in pdf_metadata.items() if v is not None})
                
                # Добавляем количество страниц
                metadata["page_count"] = len(pdf.pages)
                
        return metadata
    except Exception as e:
        logger.warning(f"Error extracting metadata from PDF file {pdf_file.name}: {e}")
        return {}

def process_pdfs_and_chunk(
    input_pdf_dir: str,
    output_md_dir: str,
    enable_formula_parsing: bool = False,
    enable_table_parsing: bool = False,
    model_source: str = "huggingface",
    models_dir: Optional[str] = None,
    backend: str = "pipeline",
    method: str = "auto",
    lang: str = "east_slavic", # Изменено значение по умолчанию
    sglang_url: Optional[str] = None,
    device: str = "cpu"
):
    """
    Основная функция обработки PDF файлов.
    
    Для каждого PDF файла в input_pdf_dir:
    1. Создает временную рабочую папку
    2. Вызывает mineru для конвертации в markdown
    3. Переносит результаты в финальную папку
    4. Очищает временные файлы
    5. Извлекает метаданные из PDF файлов

    Args:
        input_pdf_dir (str): Путь к директории с PDF файлами.
        output_md_dir (str): Путь для сохранения результатов (markdown + изображения).
        enable_formula_parsing (bool): Включить парсинг математических формул (по умолчанию False).
        enable_table_parsing (bool): Включить парсинг таблиц (по умолчанию False).
        model_source (str): Источник моделей: 'huggingface', 'modelscope' или 'local'.
        models_dir (Optional[str]): Путь к локальным моделям (только для model_source='local').
        backend (str): Бэкенд обработки: 'pipeline', 'vlm-transformers' или 'vlm-sglang-client'.
        method (str): Метод парсинга: 'auto' (определяет автоматически), 'txt' (текстовый слой) или 'ocr' (распознавание).
        lang (str): Язык для OCR (по умолчанию 'east_slavic' для восточнославянских языков).
        sglang_url (Optional[str]): URL сервера sglang (только для backend='vlm-sglang-client').
        device (str): Устройство для обработки: 'cpu' или 'cuda'.

    Raises:
        FileNotFoundError: Если входная директория не существует.
    """
    input_path = Path(input_pdf_dir)
    output_root_path = Path(output_md_dir)
    output_root_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Директория с PDF не найдена: {input_pdf_dir}")

    pdf_files = list(input_path.glob("*.pdf"))
    if not pdf_files:
        logger.info(f"В директории {input_pdf_dir} не найдено PDF файлов.")
        return

    logger.info(f"Найдено {len(pdf_files)} PDF файлов для обработки.")
    logger.info("Используется метод обработки: Subprocess (вызов команды)")

    # Настройка переменных окружения для MinerU
    env = setup_env(model_source, models_dir, enable_formula_parsing, enable_table_parsing)

    for pdf_file in pdf_files:
        temp_output_dir = None
        try:
            pdf_stem = pdf_file.stem
            logger.info(f"Обработка файла (Subprocess): {pdf_file.name}")

            # Создаем временную директорию для вывода конкретного файла
            temp_output_dir = output_root_path / f"temp_{pdf_stem}"
            final_output_dir = output_root_path / pdf_stem
            
            # Очищаем временную директорию, если она существует
            if temp_output_dir.exists():
                shutil.rmtree(temp_output_dir)
            temp_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Очищаем финальную директорию, если она существует
            if final_output_dir.exists():
                logger.info(f"  -> Удаление старой папки: {final_output_dir}")
                shutil.rmtree(final_output_dir)

            # Запуск subprocess
            result = run_subprocess(pdf_file, temp_output_dir, backend, method, lang, sglang_url, env)

            if result.returncode != 0:
                logger.error(f"Ошибка при вызове MinerU для файла {pdf_file.name}:")
                logger.error(f"Команда: {' '.join(['mineru', '-p', str(pdf_file), '-o', str(temp_output_dir), '-b', backend, '--output-format', 'md'])}")
                logger.error(f"Код возврата: {result.returncode}")
                if result.stdout:
                    logger.error(f"Stdout: {result.stdout}")
                if result.stderr:
                    logger.error(f"Stderr: {result.stderr}") 
                continue

            # Пост-обработка вывода
            if not postprocess_output(pdf_file, pdf_stem, temp_output_dir, final_output_dir, output_root_path):
                continue

            # Извлечение метаданных из PDF файла
            metadata = extract_pdf_metadata(pdf_file)
            if metadata:
                # Сохраняем метаданные в отдельный файл
                metadata_file = final_output_dir / f"{pdf_stem}_metadata.json"
                import json
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                logger.info(f"  -> Метаданные сохранены в: {metadata_file.relative_to(output_root_path)}")

        except subprocess.TimeoutExpired:
            logger.exception(f"Ошибка: Таймаут при обработке файла {pdf_file.name}.")
            continue
        except FileNotFoundError as e:
            if "mineru" in str(e).lower() or "[winerror 2]" in str(e).lower() or "No such file or directory" in str(e):
                 logger.exception(f"Команда 'mineru' не найдена. Убедитесь, что пакет 'mineru' установлен: {e}")
            else:
                 logger.exception(f"Ошибка FileNotFoundError при обработке файла {pdf_file.name}: {e}")
            continue
        except subprocess.CalledProcessError as e:
            logger.exception(f"Ошибка при вызове MinerU для файла {pdf_file.name}: {e}")
            continue
        except Exception as e:
            logger.exception(f"Неожиданная ошибка при обработке файла {pdf_file.name}: {e}")
            continue
        finally:
            # Удаляем временную директорию в любом случае
            if temp_output_dir and temp_output_dir.exists():
                shutil.rmtree(temp_output_dir)

    logger.info("Обработка PDF файлов завершена.")

