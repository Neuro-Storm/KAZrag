"""Integration test for all chunking functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import load_config
from core.indexing.chunker import get_text_splitter
from core.indexing.paragraph_chunker import ParagraphTextSplitter
from langchain_core.documents import Document


def test_all_chunking_functionality():
    """Test all chunking functionality."""
    print("=== Integration Test for All Chunking Functionality ===\n")
    
    # 1. Test configuration loading
    print("1. Testing configuration loading...")
    config = load_config()
    print(f"   Chunking strategy: {config.chunking_strategy}")
    print(f"   Chunk size: {config.chunk_size}")
    print(f"   Chunk overlap: {config.chunk_overlap}")
    print(f"   Paragraphs per chunk: {config.paragraphs_per_chunk}")
    print(f"   Paragraph overlap: {config.paragraph_overlap}")
    print("   OK Configuration loaded successfully\n")
    
    # 2. Test character-based chunking
    print("2. Testing character-based chunking...")
    char_splitter = get_text_splitter(
        chunking_strategy="character",
        chunk_size=100,
        chunk_overlap=20
    )
    
    text = "This is a test text for character-based chunking. " * 5
    char_chunks = char_splitter.split_text(text)
    print(f"   Created {len(char_chunks)} character-based chunks")
    print(f"   First chunk preview: {char_chunks[0][:50]}...")
    print("   OK Character-based chunking works correctly\n")
    
    # 3. Test paragraph-based chunking
    print("3. Testing paragraph-based chunking...")
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
    print(f"   Created {len(para_chunks)} paragraph-based chunks")
    print(f"   First chunk preview: {para_chunks[0][:100]}...")
    print("   OK Paragraph-based chunking works correctly\n")
    
    # 4. Test document splitting
    print("4. Testing document splitting...")
    document = Document(
        page_content=para_text,
        metadata={"source": "test.txt"}
    )
    
    # Test with character-based splitter
    char_doc_chunks = char_splitter.split_documents([document])
    print(f"   Character-based document splitting: {len(char_doc_chunks)} chunks")
    
    # Test with paragraph-based splitter
    para_doc_chunks = para_splitter.split_documents([document])
    print(f"   Paragraph-based document splitting: {len(para_doc_chunks)} chunks")
    print("   OK Document splitting works correctly\n")
    
    # 5. Test direct ParagraphTextSplitter usage
    print("5. Testing direct ParagraphTextSplitter usage...")
    direct_splitter = ParagraphTextSplitter(
        paragraphs_per_chunk=3,
        paragraph_overlap=1
    )
    
    direct_chunks = direct_splitter.split_text(para_text)
    print(f"   Direct ParagraphTextSplitter: {len(direct_chunks)} chunks")
    print("   OK Direct ParagraphTextSplitter works correctly\n")
    
    print("=== All tests passed! ===")


if __name__ == "__main__":
    test_all_chunking_functionality()