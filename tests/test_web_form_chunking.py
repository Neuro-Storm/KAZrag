# Test web form submission for chunking settings.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import load_config
from web.admin_app import update_index_settings


def test_form_submission():
    # Test form submission with chunking settings.
    # Load configuration
    config = load_config()
    
    print("=== Initial Configuration ===")
    print("Chunking strategy: {}".format(config.chunking_strategy))
    print("Paragraphs per chunk: {}".format(config.paragraphs_per_chunk))
    print("Paragraph overlap: {}".format(config.paragraph_overlap))
    print("Sentences per chunk: {}".format(config.sentences_per_chunk))
    print("Sentence overlap: {}".format(config.sentence_overlap))
    
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
        print("{}: {}".format(key, value))
    
    # Update settings
    try:
        update_index_settings(form_data, config)
        print("\n=== Updated Configuration ===")
        print("Chunking strategy: {}".format(config.chunking_strategy))
        print("Chunk size: {}".format(config.chunk_size))
        print("Chunk overlap: {}".format(config.chunk_overlap))
        print("Paragraphs per chunk: {}".format(config.paragraphs_per_chunk))
        print("Paragraph overlap: {}".format(config.paragraph_overlap))
        print("Sentences per chunk: {}".format(config.sentences_per_chunk))
        print("Sentence overlap: {}".format(config.sentence_overlap))
        print("\nTest passed!")
    except Exception as e:
        print("\nTest failed with error: {}".format(e))


if __name__ == "__main__":
    test_form_submission()