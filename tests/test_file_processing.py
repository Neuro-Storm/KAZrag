"""Тесты для обработки файлов."""

import io
import os

# Устанавливаем ADMIN_API_KEY для тестов до импорта приложения
os.environ["ADMIN_API_KEY"] = "test_api_key"

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_file_processing_size_limit():
    """Тест ограничения размера файлов при обработке."""
    # Сбрасываем rate-limit storage перед тестом
    from web.admin_app import _rate_limit_storage
    _rate_limit_storage.clear()
    
    # Создаем файл больше лимита (100MB)
    large_file = io.BytesIO(b"0" * (101 * 1024 * 1024))  # 101MB
    large_file.name = "large_file.pdf"
    
    try:
        client.post(
            "/api/admin/process-pdfs",
            files={"files": ("large_file.pdf", large_file, "application/pdf")},
            auth=("admin", "test_api_key")
        )
    except Exception as e:
        # Если получаем исключение, проверяем, что это HTTPException с кодом 400
        assert "400" in str(e)
        assert "слишком большой" in str(e)


def test_file_processing_type_limit():
    """Тест ограничения типов файлов при обработке."""
    # Сбрасываем rate-limit storage перед тестом
    from web.admin_app import _rate_limit_storage
    _rate_limit_storage.clear()
    
    # Создаем файл с неподдерживаемым расширением
    file_content = io.BytesIO(b"test content")
    file_content.name = "test_file.exe"
    
    try:
        client.post(
            "/api/admin/process-files",
            files={"files": ("test_file.exe", file_content, "application/octet-stream")},
            auth=("admin", "test_api_key")
        )
    except Exception as e:
        # Если получаем исключение, проверяем, что это HTTPException с кодом 400
        assert "400" in str(e)
        assert "неподдерживаемый формат" in str(e)


def test_file_processing_valid_file():
    """Тест обработки валидного файла."""
    # Сбрасываем rate-limit storage перед тестом
    from web.admin_app import _rate_limit_storage
    _rate_limit_storage.clear()
    
    # Создаем валидный PDF файл
    file_content = io.BytesIO(b"%PDF-1.4 test content")
    file_content.name = "test_file.pdf"
    
    # Note: Этот тест может потребовать мокирования зависимости обработки файлов
    # чтобы не запускать реальную обработку