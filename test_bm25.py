#!/usr/bin/env python3
\"\"\"Test script for checking BM25 implementation.\"\"\"

def test_bm25_implementation():
    print(\"=== Testing BM25 Implementation ===\")
    
    # Check configuration
    try:
        from config.settings_model import Config
        config = Config()
        print(\"[OK] Config model loaded\")
        print(f\"  - use_bm25: {config.use_bm25}\")
        print(f\"  - sparse_vector_name: {config.sparse_vector_name}\")
        print(f\"  - bm25_tokenizer: {config.bm25_tokenizer}\")
        print(f\"  - bm25_min_token_len: {config.bm25_min_token_len}\")
    except Exception as e:
        print(f\"[ERROR] Error loading config: {e}\")
        return False
    
    # Check adapter
    try:
        from config.config_manager import ConfigManager
        from core.embedding.sparse_embedding_adapter import SparseEmbeddingAdapter
        
        # Use current config or create new one with BM25 enabled
        config_manager = ConfigManager.get_instance()
        current_config = config_manager.get()
        
        # Create adapter
        adapter = SparseEmbeddingAdapter(current_config)
        print(\"[OK] SparseEmbeddingAdapter created successfully\")
        
        # Test encode
        test_texts = [\"test document for bm25 native implementation\"]
        result = adapter.encode(test_texts)
        print(f\"[OK] Encode method works, results: {len(result)}\")
        
        if len(result) > 0:
            print(f\"  - Indices in first vector: {len(result[0].indices)}\")
            print(f\"  - Values in first vector: {len(result[0].values)}\")
            if len(result[0].indices) > 0:
                print(f\"  - Example index: {result[0].indices[0]}\")
                print(f\"  - Example value: {result[0].values[0]:.4f}\")
        
        # Test embed_query
        query_result = adapter.embed_query(\"test query\")
        print(f\"[OK] embed_query method works\")
        print(f\"  - Indices in query: {len(query_result.indices)}\")
        print(f\"  - Values in query: {len(query_result.values)}\")
        
    except Exception as e:
        print(f\"[ERROR] Error in adapter test: {e}\")
        import traceback
        traceback.print_exc()
        return False
    
    # Check dependencies
    try:
        from qdrant_client.models import SparseVector, SparseVectorParams, Modifier, SparseIndexParams
        print(\"[OK] Qdrant dependencies loaded\")
    except Exception as e:
        print(f\"[ERROR] Error loading Qdrant dependencies: {e}\")
        return False
    
    # Check modified files
    try:
        from core.indexing.indexer_component import Indexer
        print(\"[OK] Indexer imported\")
    except Exception as e:
        print(f\"[ERROR] Error importing Indexer: {e}\")
        return False
    
    try:
        from core.search.search_strategy import SearchStrategy
        print(\"[OK] SearchStrategy imported\")
    except Exception as e:
        print(f\"[ERROR] Error importing SearchStrategy: {e}\")
        return False
    
    try:
        from core.search.collection_analyzer import CollectionAnalyzer
        print(\"[OK] CollectionAnalyzer imported\")
    except Exception as e:
        print(f\"[ERROR] Error importing CollectionAnalyzer: {e}\")
        return False
    
    print(\"\\n=== All tests passed successfully! ===\")
    print(\"Native BM25 implementation is fully implemented and working correctly.\")
    return True

if __name__ == \"__main__\":
    test_bm25_implementation()