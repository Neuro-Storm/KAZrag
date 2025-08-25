from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Попробуем создать TestClient
try:
    client = TestClient(app)
    print("TestClient(app) - работает")
    
    # Попробуем сделать запрос
    response = client.get("/")
    print(f"Ответ: {response.json()}")
except Exception as e:
    print(f"TestClient(app) - ошибка: {e}")