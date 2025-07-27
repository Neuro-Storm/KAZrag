
"""
Модуль для преобразования PDF в Markdown с помощью утилиты MinerU.

Основные функции:
- Конвертирует PDF файлы в Markdown формат
- Сохраняет изображения в отдельную папку
- Поддерживает различные настройки парсинга (формулы, таблицы)
- Обрабатывает файлы через subprocess вызов mineru

Структура выходных данных:
output_md_dir/
  ├── filename/
  │   ├── filename.md       # Основной markdown файл
  │   └── images/           # Папка с извлеченными изображениями
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional

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
        print(f"В директории {input_pdf_dir} не найдено PDF файлов.")
        return

    print(f"Найдено {len(pdf_files)} PDF файлов для обработки.")
    print("Используется метод обработки: Subprocess (вызов команды)")

    # Настройка переменных окружения для MinerU
    env = os.environ.copy()
    env["MINERU_MODEL_SOURCE"] = model_source
    if models_dir and model_source == "local":
        env["MINERU_MODELS_DIR"] = models_dir

    # Настройка флагов парсинга формул и таблиц через переменные окружения
    env["MINERU_ENABLE_FORMULA_PARSING"] = str(enable_formula_parsing).lower()
    env["MINERU_ENABLE_TABLE_PARSING"] = str(enable_table_parsing).lower()

    for pdf_file in pdf_files:
        try:
            pdf_stem = pdf_file.stem
            print(f"Обработка файла (Subprocess): {pdf_file.name}")

            # Создаем временную директорию для вывода конкретного файла
            temp_output_dir = output_root_path / f"temp_{pdf_stem}"
            final_output_dir = output_root_path / pdf_stem
            
            # Очищаем временную директорию, если она существует
            if temp_output_dir.exists():
                shutil.rmtree(temp_output_dir)
            temp_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Очищаем финальную директорию, если она существует
            if final_output_dir.exists():
                print(f"  -> Удаление старой папки: {final_output_dir}")
                shutil.rmtree(final_output_dir)

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

            print(f"  -> Выполняется команда: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=600
            )

            if result.returncode != 0:
                print(f"Ошибка при вызове MinerU для файла {pdf_file.name}:")
                print(f"Команда: {' '.join(cmd)}")
                print(f"Код возврата: {result.returncode}")
                if result.stdout:
                    print(f"Stdout: {result.stdout}")
                if result.stderr:
                    print(f"Stderr: {result.stderr}") 
                # Удаляем временную папку в случае ошибки
                if temp_output_dir.exists():
                    shutil.rmtree(temp_output_dir)
                continue

            # Пост-обработка: перенос результатов из временной папки в финальную
            # Ищем созданный .md файл в temp_output_dir или подпапках
            md_files = list(temp_output_dir.rglob("*.md"))
            if not md_files:
                print(f"  -> Предупреждение: Не найден .md файл после обработки {pdf_file.name}.")
                if temp_output_dir.exists():
                    shutil.rmtree(temp_output_dir)
                continue

            # Предполагаем, что основной файл имеет имя, совпадающее с именем PDF
            target_md_file = temp_output_dir / f"{pdf_stem}.md"
            if not target_md_file.exists():
                 # Если нет, берем первый найденный
                 target_md_file = md_files[0]
                 print(f"  -> Используется найденный файл: {target_md_file.relative_to(temp_output_dir)}")

            # Создаем финальную папку для этого PDF
            final_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Перемещаем .md файл
            final_md_path = final_output_dir / f"{pdf_stem}.md"
            target_md_file.rename(final_md_path)
            print(f"  -> .md файл перемещен в: {final_md_path.relative_to(output_root_path)}")
            
            # Перемещаем папку с изображениями, если она есть
            images_dir_in_temp = temp_output_dir / "images"
            if images_dir_in_temp.exists() and images_dir_in_temp.is_dir():
                images_dir_final = final_output_dir / "images"
                # shutil.copytree может потребоваться, если нужно сохранить оригиналы
                shutil.move(str(images_dir_in_temp), str(images_dir_final)) 
                print(f"  -> Папка 'images' перемещена в: {images_dir_final.relative_to(output_root_path)}")
            
            # Удаляем временную директорию
            if temp_output_dir.exists():
                shutil.rmtree(temp_output_dir)

        except subprocess.TimeoutExpired:
            print(f"Ошибка: Таймаут при обработке файла {pdf_file.name}.")
            # Очищаем временную папку в случае таймаута
            temp_dir = output_root_path / f"temp_{pdf_stem}"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            continue
        except FileNotFoundError as e:
            if "mineru" in str(e).lower() or "[winerror 2]" in str(e).lower() or "No such file or directory" in str(e):
                 print(f"Ошибка: Команда 'mineru' не найдена. Убедитесь, что пакет 'mineru' установлен.")
                 print(f"Подробности ошибки: {e}")
            else:
                 print(f"Ошибка FileNotFoundError при обработке файла {pdf_file.name}: {e}")
            # Очищаем временную папку в случае критической ошибки
            temp_dir = output_root_path / f"temp_{pdf_stem}"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            break
        except Exception as e:
            print(f"Неожиданная ошибка при обработке файла {pdf_file.name}: {e}")
            import traceback
            traceback.print_exc()
            # Очищаем временную папку в случае ошибки
            temp_dir = output_root_path / f"temp_{pdf_stem}"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            continue

    print("Обработка PDF файлов завершена.")

