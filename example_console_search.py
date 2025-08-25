"""
Example of using the KAZrag console search functionality programmatically.
This demonstrates how to integrate the search functionality into other applications.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from config.settings import load_config
from core.searcher import search_in_collection
from core.qdrant_client import get_qdrant_client


async def example_search():
    """Example of performing a search programmatically."""
    try:
        # Load configuration
        config = load_config()
        
        # Initialize Qdrant client
        client = get_qdrant_client(config)
        
        # Perform search
        query = "Пример поискового запроса"
        collection_name = config.collection_name
        device = "cpu"  # or "cuda"
        k = 5  # number of results
        
        print(f"Searching for: {query}")
        print(f"Collection: {collection_name}")
        print(f"Device: {device}")
        print(f"Results count: {k}")
        
        results, error = await search_in_collection(
            query=query,
            collection_name=collection_name,
            device=device,
            k=k,
            client=client
        )
        
        if error:
            print(f"Error: {error}")
            return
            
        print(f"\nFound {len(results)} results:")
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
                content = combined_metadata.get('page_content', 'No content')
            
            source = combined_metadata.get('source', 'Unknown')
            
            print(f"\n{i}. Source: {source}")
            print(f"   Score: {score:.4f}")
            print(f"   Content: {content[:100]}...")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(example_search())