"""Тест для проверки доступности страницы индексации."""
import requests
from requests.auth import HTTPBasicAuth
import os

def test_indexing_page():
    """Тестирует доступ к странице индексации."""
    base_url = "http://localhost:8000"
    indexing_endpoint = "/api/indexing/"
    full_url = base_url + indexing_endpoint
    
    print(f"Проверяем доступ к: {full_url}")
    
    # Получаем ADMIN_API_KEY из переменных окружения
    admin_api_key = os.getenv("ADMIN_API_KEY")
    
    try:
        if admin_api_key:
            # Если API ключ установлен, используем HTTP Basic Auth
            print("Обнаружен ADMIN_API_KEY, используем аутентификацию...")
            response = requests.get(
                full_url,
                auth=HTTPBasicAuth("admin", admin_api_key),  # имя пользователя может быть любым
                timeout=10
            )
        else:
            # Если API ключ не установлен, делаем обычный запрос
            print("ADMIN_API_KEY не установлен, делаем запрос без аутентификации...")
            response = requests.get(full_url, timeout=10)
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            print("[SUCCESS] Страница индексации доступна")
            print("Заголовок страницы (первые 100 символов):", response.text[:100])
        elif response.status_code == 401:
            print("[ERROR] Требуется аутентификация. Проверьте значение ADMIN_API_KEY в .env файле.")
        elif response.status_code == 404:
            print("[ERROR] Страница не найдена. Проверьте маршруты и запущено ли приложение.")
        elif response.status_code == 500:
            print("[ERROR] Внутренняя ошибка сервера. Проверьте логи приложения.")
            print("Ответ сервера:", response.text[:500])
        else:
            print(f"[ERROR] Неожиданный статус код: {response.status_code}")
            print("Ответ сервера:", response.text[:500])
            
        return response.status_code
        
    except requests.exceptions.ConnectionError:
        print("[ERROR] Не удалось подключиться к серверу. Убедитесь, что приложение запущено на http://localhost:8000")
        return None
    except requests.exceptions.Timeout:
        print("[ERROR] Таймаут запроса. Сервер не отвечает в течение 10 секунд.")
        return None
    except Exception as e:
        print(f"[ERROR] Произошла ошибка при выполнении запроса: {e}")
        return None

def test_routes():
    """Тестирует основные маршруты приложения."""
    base_url = "http://localhost:8000"
    
    routes = [
        "/",
        "/api/search/",
        "/api/admin/settings/",
        "/api/indexing/"
    ]
    
    admin_api_key = os.getenv("ADMIN_API_KEY")
    auth = HTTPBasicAuth("admin", admin_api_key) if admin_api_key else None
    
    print("\nПроверяем основные маршруты:")
    for route in routes:
        url = base_url + route
        
        try:
            if route.startswith("/api/admin/") or route.startswith("/api/indexing/"):
                # Для этих маршрутов может потребоваться аутентификация
                response = requests.get(url, auth=auth, timeout=10) if auth else requests.get(url, timeout=10)
            else:
                response = requests.get(url, timeout=10)
                
            status_msg = "[SUCCESS]" if response.status_code == 200 else "[ERROR]"
            print(f"{status_msg} {route} -> {response.status_code}")
        except Exception as e:
            print(f"[ERROR] {route} -> Ошибка: {e}")

if __name__ == "__main__":
    print("Тестирование доступности страницы индексации...")
    status_code = test_indexing_page()
    
    print("\n" + "="*50)
    test_routes()
    
    if status_code == 200:
        print("\n[SUCCESS] Тест пройден успешно!")
    else:
        print(f"\n[ERROR] Тест завершился со статусом: {status_code}")