"""Integration test for different chunking strategies."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.indexing.chunker import get_text_splitter


def test_chunking_strategies():
    """Test different chunking strategies."""
    # Test character-based chunking
    print("=== Character-based chunking ===")
    char_splitter = get_text_splitter(
        chunking_strategy="character",
        chunk_size=100,
        chunk_overlap=20
    )
    
    text = "This is a test text for chunking. " * 10  # Повторяем текст для демонстрации чанкинга
    char_chunks = char_splitter.split_text(text)
    print(f"Character-based chunking produced {len(char_chunks)} chunks")
    for i, chunk in enumerate(char_chunks[:2]):  # Покажем только первые два чанка
        print(f"Chunk {i+1}: {repr(chunk[:50])}...")
    
    # Test paragraph-based chunking
    print("\n=== Paragraph-based chunking ===")
    para_splitter = get_text_splitter(
        chunking_strategy="paragraph",
        paragraphs_per_chunk=2,
        paragraph_overlap=1
    )
    
    para_text = """First paragraph of text for testing paragraph chunking.

Second paragraph of text for testing paragraph chunking.

Third paragraph of text for testing paragraph chunking.

Fourth paragraph of text for testing paragraph chunking."""
    
    para_chunks = para_splitter.split_text(para_text)
    print(f"Paragraph-based chunking produced {len(para_chunks)} chunks")
    for i, chunk in enumerate(para_chunks):
        print(f"Chunk {i+1}: {repr(chunk[:100])}...")


if __name__ == "__main__":
    test_chunking_strategies()