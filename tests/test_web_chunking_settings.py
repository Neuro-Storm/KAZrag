"""Test for web interface chunking settings."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import load_config


def test_web_chunking_settings():
    """Test web interface chunking settings."""
    # Load configuration
    config = load_config()
    
    print("=== Web Interface Chunking Settings Test ===")
    print(f"Chunking strategy: {config.chunking_strategy}")
    print(f"Chunk size: {config.chunk_size}")
    print(f"Chunk overlap: {config.chunk_overlap}")
    print(f"Paragraphs per chunk: {config.paragraphs_per_chunk}")
    print(f"Paragraph overlap: {config.paragraph_overlap}")
    
    # Test that all required fields are present
    required_fields = [
        'chunking_strategy',
        'chunk_size',
        'chunk_overlap',
        'paragraphs_per_chunk',
        'paragraph_overlap'
    ]
    
    for field in required_fields:
        if hasattr(config, field):
            print(f"OK Field '{field}' is present")
        else:
            print(f"ERROR Field '{field}' is missing")
    
    # Test default values
    print("\n=== Default Values Test ===")
    print(f"Default chunking strategy: {config.chunking_strategy}")
    print(f"Default paragraphs per chunk: {config.paragraphs_per_chunk}")
    print(f"Default paragraph overlap: {config.paragraph_overlap}")


if __name__ == "__main__":
    test_web_chunking_settings()