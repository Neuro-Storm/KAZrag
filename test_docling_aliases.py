#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Тест для проверки алиасов, используемых в docling_converter.py
"""
import sys
import os

# Добавляем путь к проекту
sys.path.insert(0, os.path.abspath('.'))

def test_docling_converter_aliases():
    print("Тестируем алиасы, используемые в docling_converter.py...")
    
    try:
        from config.settings import Config
        config = Config()
        print("[+] Импорт и создание Config прошло успешно")
    except Exception as e:
        print(f"[-] Ошибка при импорте/создании Config: {e}")
        return False

    # Список алиасов, используемых в docling_converter.py
    aliases_to_test = [
        ('docling_backend', 'main.docling.backend'),
        ('docling_device', 'main.docling.device'),
        ('docling_ocr_lang', 'main.docling.ocr_lang'),
        ('docling_use_ocr', 'main.docling.use_ocr'),
        ('docling_use_tables', 'main.docling.use_tables'),
        ('huggingface_cache_path', 'main.model_paths.huggingface_cache_path'),
        ('granite_models_dir', 'main.docling.granite_models_dir'),
    ]

    all_passed = True
    for alias_name, direct_path in aliases_to_test:
        try:
            # Получаем значение через алиас
            alias_value = getattr(config, alias_name)
            
            # Получаем значение через прямой доступ
            parts = direct_path.split('.')
            direct_value = config
            for part in parts:
                direct_value = getattr(direct_value, part)
            
            # Проверяем, что значения совпадают
            if alias_value == direct_value:
                print(f"[+] {alias_name} = {alias_value} (совпадает с {direct_path})")
            else:
                print(f"[-] {alias_name} != {direct_path}: {alias_value} != {direct_value}")
                all_passed = False
        except Exception as e:
            print(f"[-] Ошибка доступа к {alias_name} или {direct_path}: {e}")
            all_passed = False

    if all_passed:
        print("\nВсе тесты алиасов для docling_converter прошли успешно!")
        return True
    else:
        print("\nНе все тесты алиасов прошли успешно!")
        return False

if __name__ == "__main__":
    success = test_docling_converter_aliases()
    if not success:
        sys.exit(1)