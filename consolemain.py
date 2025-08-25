"""Console interface for KAZrag search application."""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

# Import required modules
from config.settings import load_config, Config
from core.searcher import search_in_collection
from core.qdrant_client import get_qdrant_client
from core.qdrant_collections import get_cached_collections
from logging_config import setup_logging

logger = logging.getLogger(__name__)


def print_results(results, error=None):
    """Print search results in a formatted way."""
    if error:
        print(f"Ошибка при поиске: {error}")
        return
    
    if not results:
        print("Ничего не найдено.")
        return
    
    print(f"\nНайдено {len(results)} результатов:")
    print("-" * 80)
    
    for i, (result, score) in enumerate(results, 1):
        # Extract metadata from result
        # Handle both regular dictionaries and Pydantic Document objects
        if hasattr(result, 'metadata'):
            metadata = result.metadata or {}
        else:
            metadata = result.get('metadata', {}) if isinstance(result, dict) else {}
            
        if hasattr(result, 'payload'):
            payload = result.payload or {}
        else:
            payload = result.get('payload', {}) if isinstance(result, dict) else {}
        
        # Combine metadata and payload
        combined_metadata = {**metadata, **payload}
        
        # Get document content
        # For Document objects, content is in page_content attribute
        # For dict objects, it might be in page_content field of metadata/payload
        if hasattr(result, 'page_content'):
            content = result.page_content
        else:
            content = combined_metadata.get('page_content', 'Нет содержимого')
        
        # Get document info
        doc_id = combined_metadata.get('id', 'Неизвестно')
        source = combined_metadata.get('source', 'Неизвестно')
        author = combined_metadata.get('author', 'Неизвестно')
        
        print(f"\n{i}. Документ: {source}")
        print(f"   Автор: {author}")
        print(f"   ID: {doc_id}")
        print(f"   Релевантность: {score:.4f}")
        print(f"   Содержимое: {content[:200]}{'...' if len(content) > 200 else ''}")
        print("-" * 80)


def get_user_input(config: Config, collections: list):
    """Get search parameters from user."""
    print("\nПараметры поиска:")
    
    # Get query
    query = input("Введите поисковый запрос: ").strip()
    if not query:
        print("Запрос не может быть пустым.")
        return None
    
    # Select collection
    if not collections:
        print("Нет доступных коллекций.")
        return None
        
    print("\nДоступные коллекции:")
    for i, collection in enumerate(collections, 1):
        print(f"{i}. {collection}")
    
    if len(collections) == 1:
        selected_collection = collections[0]
        print(f"Выбрана коллекция: {selected_collection}")
    else:
        try:
            choice = int(input(f"Выберите коллекцию (1-{len(collections)}): ")) - 1
            if 0 <= choice < len(collections):
                selected_collection = collections[choice]
            else:
                print("Неверный выбор.")
                return None
        except ValueError:
            print("Неверный ввод.")
            return None
    
    # Get number of results
    k_input = input(f"Количество результатов (по умолчанию {config.search_default_k}): ").strip()
    k = config.search_default_k
    if k_input:
        try:
            k = int(k_input)
        except ValueError:
            print("Неверное значение, используется значение по умолчанию.")
    
    # Get search device
    print("\nДоступные устройства для поиска:")
    print("1. cpu")
    print("2. cuda")
    device_choice = input("Выберите устройство (1-2, по умолчанию cpu): ").strip()
    device = "cpu"
    if device_choice == "2":
        device = "cuda"
    elif device_choice and device_choice not in ["1", "2"]:
        print("Неверный выбор, используется значение по умолчанию.")
    
    # Get search type
    print("\nТип поиска:")
    print("1. Плотный (dense)")
    print("2. Гибридный (hybrid)")
    print("3. Разреженный (sparse)")
    search_type_choice = input("Выберите тип поиска (1-3, по умолчанию плотный): ").strip()
    hybrid = False
    if search_type_choice == "2":
        hybrid = True
    elif search_type_choice == "3":
        # For sparse search, we'll set hybrid to False but the searcher will handle it
        pass
    elif search_type_choice and search_type_choice not in ["1", "2", "3"]:
        print("Неверный выбор, используется значение по умолчанию.")
    
    return {
        "query": query,
        "collection": selected_collection,
        "device": device,
        "k": k,
        "hybrid": hybrid
    }


async def run_search(config: Config, client, search_params: dict):
    """Run the search with given parameters."""
    try:
        print("\nВыполняется поиск...")
        results, error = await search_in_collection(
            query=search_params["query"],
            collection_name=search_params["collection"],
            device=search_params["device"],
            k=search_params["k"],
            hybrid=search_params["hybrid"],
            client=client
        )
        print_results(results, error)
        return True
    except Exception as e:
        logger.exception(f"Ошибка при выполнении поиска: {e}")
        print(f"Ошибка при выполнении поиска: {e}")
        return False


async def main():
    """Main console interface function."""
    print("Добро пожаловать в консольный интерфейс KAZrag!")
    print("=" * 50)
    
    try:
        # Setup logging
        setup_logging()
        
        # Load configuration
        print("Загрузка конфигурации...")
        config = load_config()
        print("Конфигурация загружена.")
        
        # Initialize Qdrant client
        print("Подключение к Qdrant...")
        client = get_qdrant_client(config)
        print("Подключение к Qdrant установлено.")
        
        # Get available collections
        print("Получение списка коллекций...")
        collections = get_cached_collections(client=client)
        if collections:
            print(f"Доступные коллекции: {', '.join(collections)}")
        else:
            print("Нет доступных коллекций.")
        
        while True:
            # Get search parameters from user
            search_params = get_user_input(config, collections)
            if not search_params:
                continue
            
            # Run search
            success = await run_search(config, client, search_params)
            
            # Ask if user wants to continue
            if success:
                continue_choice = input("\nПродолжить поиск? (y/n, по умолчанию y): ").strip().lower()
                if continue_choice == "n":
                    break
            else:
                retry_choice = input("\nПопробовать снова? (y/n, по умолчанию y): ").strip().lower()
                if retry_choice == "n":
                    break
                    
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана пользователем.")
    except Exception as e:
        logger.exception(f"Ошибка приложения: {e}")
        print(f"Ошибка приложения: {e}")
        return 1
    
    print("\nСпасибо за использование KAZrag!")
    return 0


if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)