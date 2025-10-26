#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Тест для проверки алиасов, упомянутых в отчете об ошибках
"""
import sys
import os

# Добавляем путь к проекту
sys.path.insert(0, os.path.abspath('.'))

def test_aliases_from_error_report():
    print("Тестируем алиасы, упомянутые в отчете об ошибках...")
    
    try:
        from config.settings import Config
        config = Config()
        print("[+] Импорт и создание Config прошло успешно")
    except Exception as e:
        print(f"[-] Ошибка при импорте/создании Config: {e}")
        return False

    # Проверяем use_bm25 (основная ошибка из отчета)
    try:
        use_bm25_value = config.use_bm25
        print(f"[+] config.use_bm25 = {use_bm25_value}")
    except Exception as e:
        print(f"[-] Ошибка доступа к config.use_bm25: {e}")
        return False

    # Проверяем docling_backend (упомянут в отчете)
    try:
        docling_backend_value = config.docling_backend
        print(f"[+] config.docling_backend = {docling_backend_value}")
    except Exception as e:
        print(f"[-] Ошибка доступа к config.docling_backend: {e}")
        return False

    # Проверяем granite_models_dir (упомянут в startup.py)
    try:
        granite_models_dir_value = config.granite_models_dir
        print(f"[+] config.granite_models_dir = {granite_models_dir_value}")
    except Exception as e:
        print(f"[-] Ошибка доступа к config.granite_models_dir: {e}")
        return False

    # Проверим, что значения соответствуют ожидаемым вложенным путям
    try:
        if config.use_bm25 == config.main.bm25.enabled:
            print("[+] config.use_bm25 соответствует config.main.bm25.enabled")
        else:
            print(f"[-] config.use_bm25 != config.main.bm25.enabled: {config.use_bm25} != {config.main.bm25.enabled}")
            return False
    except Exception as e:
        print(f"[-] Ошибка проверки соответствия use_bm25: {e}")
        return False

    try:
        if config.docling_backend == config.main.docling.backend:
            print("[+] config.docling_backend соответствует config.main.docling.backend")
        else:
            print(f"[-] config.docling_backend != config.main.docling.backend: {config.docling_backend} != {config.main.docling.backend}")
            return False
    except Exception as e:
        print(f"[-] Ошибка проверки соответствия docling_backend: {e}")
        return False

    try:
        if config.granite_models_dir == config.main.docling.granite_models_dir:
            print("[+] config.granite_models_dir соответствует config.main.docling.granite_models_dir")
        else:
            print(f"[-] config.granite_models_dir != config.main.docling.granite_models_dir: {config.granite_models_dir} != {config.main.docling.granite_models_dir}")
            return False
    except Exception as e:
        print(f"[-] Ошибка проверки соответствия granite_models_dir: {e}")
        return False

    print("\nВсе тесты алиасов пройдены успешно!")
    return True

if __name__ == "__main__":
    success = test_aliases_from_error_report()
    if not success:
        sys.exit(1)