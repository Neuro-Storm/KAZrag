import asyncio
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

try:
    from config.config_manager import ConfigManager
    from core.qdrant.qdrant_client import get_qdrant_client
    from core.search.searcher import search_in_collection
    print("OK: All modules imported successfully")
except Exception as e:
    print(f"ERROR: Failed to import modules: {e}")
    sys.exit(1)

async def test_search():
    try:
        # Load configuration
        config_manager = ConfigManager.get_instance()
        config = config_manager.get()
        print(f"OK: Configuration loaded, collection: {config.collection_name}")
        
        # Initialize Qdrant client
        client = get_qdrant_client(config)
        print("OK: Qdrant client initialized")
        
        # Test search
        query = "test query"
        results, error = await search_in_collection(
            query=query,
            collection_name=config.collection_name,
            device="cpu",
            k=5,
            hybrid=False,
            client=client
        )
        
        if error:
            print(f"ERROR: Search error: {error}")
        else:
            print(f"OK: Search completed successfully, found {len(results)} results")
            
    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_search())