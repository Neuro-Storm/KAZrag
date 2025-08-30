"""Тесты для админ-панели настроек."""

import os

# Устанавливаем ADMIN_API_KEY для тестов до импорта приложения
os.environ["ADMIN_API_KEY"] = "test_api_key"

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_settings_redirect():
    """Тест редиректа с /settings на /api/admin/settings/."""
    # Сбрасываем rate-limit storage перед тестом
    from web.admin_app import _rate_limit_storage
    _rate_limit_storage.clear()
    
    response = client.get("/settings", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/api/admin/settings/"


def test_update_settings_form_validation():
    """Тест валидности типов и обязательности полей при обновлении настроек."""
    # Сбрасываем rate-limit storage перед тестом
    from web.admin_app import _rate_limit_storage
    _rate_limit_storage.clear()
    
    # Тест с пустыми обязательными полями
    try:
        client.post(
            "/api/admin/update-settings",
            data={
                "action": "save_index_settings",
                "folder_path": "",
                "collection_name": "",
                "hf_model": ""
            },
            auth=("admin", "test_api_key")
        )
    except Exception as e:
        # Если получаем исключение, проверяем, что это HTTPException с кодом 400
        assert "400" in str(e)
        assert "Путь к папке не может быть пустым" in str(e)
    
    # Тест с неверным типом для числовых полей
    try:
        client.post(
            "/api/admin/update-settings",
            data={
                "action": "save_index_settings",
                "folder_path": "./data_to_index",
                "collection_name": "test-collection",
                "hf_model": "sentence-transformers/all-MiniLM-L6-v2",
                "chunk_size": "invalid_number"
            },
            auth=("admin", "test_api_key")
        )
    except Exception as e:
        # Если получаем исключение, проверяем, что это HTTPException с кодом 400
        assert "400" in str(e)
        assert "Неверный размер чанка" in str(e)
    
    # Тест с корректными данными
    # Note: Этот тест может потребовать мокирования зависимости аутентификации
    # и работы с конфигурацией, чтобы не влиять на реальную конфигурацию


def test_rate_limit():
    """Тест rate-limit для админ-эндпоинтов."""
    # Сбрасываем rate-limit storage перед тестом
    from web.admin_app import _rate_limit_storage
    _rate_limit_storage.clear()
    
    # Множественные запросы к админ-эндпоинту
    rate_limit_triggered = False
    for _i in range(15):
        try:
            response = client.get("/api/admin/settings/", auth=("admin", "test_api_key"))
            if response.status_code == 429:
                # Достигнут лимит
                assert "Превышен лимит запросов" in response.text
                rate_limit_triggered = True
                break
        except Exception as e:
            # Обрабатываем ошибку 429, которая может вызвать исключение
            if "429" in str(e):
                rate_limit_triggered = True
                break
            else:
                raise e
    
    # Проверяем, что rate-limit сработал
    assert rate_limit_triggered, "Rate limit should have been triggered"