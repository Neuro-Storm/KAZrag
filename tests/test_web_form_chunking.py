# Test web form submission for chunking settings.

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import load_config
from web.admin_app import update_index_settings


def test_form_submission():
    # Test form submission with chunking settings.
    # Load configuration
    config = load_config()
    
    print("=== Initial Configuration ===")
    print(f"Chunking strategy: {config.chunking_strategy}")
    print(f"Paragraphs per chunk: {config.paragraphs_per_chunk}")
    print(f"Paragraph overlap: {config.paragraph_overlap}")
    print(f"Sentences per chunk: {config.sentences_per_chunk}")
    print(f"Sentence overlap: {config.sentence_overlap}")
    
    # Simulate form data with all required fields
    form_data = {
        "folder_path": "./data_to_index",
        "collection_name": "test-collection",
        "hf_model": "sentence-transformers/all-MiniLM-L6-v2",
        "chunking_strategy": "sentence",
        "chunk_size": "300",
        "chunk_overlap": "50",
        "paragraphs_per_chunk": "2",
        "paragraph_overlap": "1",
        "sentences_per_chunk": "3",
        "sentence_overlap": "1",
        "embedding_batch_size": "32",
        "indexing_batch_size": "50",
        "use_dense": False,
        "device": "auto"
    }
    
    print("\n=== Form Data ===")
    for key, value in form_data.items():
        print(f"{key}: {value}")
    
    # Update settings
    try:
        update_index_settings(form_data, config)
        print("\n=== Updated Configuration ===")
        print(f"Chunking strategy: {config.chunking_strategy}")
        print(f"Chunk size: {config.chunk_size}")
        print(f"Chunk overlap: {config.chunk_overlap}")
        print(f"Paragraphs per chunk: {config.paragraphs_per_chunk}")
        print(f"Paragraph overlap: {config.paragraph_overlap}")
        print(f"Sentences per chunk: {config.sentences_per_chunk}")
        print(f"Sentence overlap: {config.sentence_overlap}")
        print("\nTest passed!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")


if __name__ == "__main__":
    test_form_submission()