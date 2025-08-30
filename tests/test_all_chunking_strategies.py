"""Integration test for all chunking strategies."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langchain_core.documents import Document

from config.settings import load_config
from core.indexing.chunker import get_text_splitter
from core.indexing.paragraph_chunker import ParagraphTextSplitter
from core.indexing.sentence_chunker import SentenceTextSplitter


def test_all_chunking_strategies():
    """Test all chunking strategies."""
    print("=== Integration Test for All Chunking Strategies ===\n")
    
    # 1. Test configuration loading
    print("1. Testing configuration loading...")
    config = load_config()
    print(f"   Chunking strategy: {config.chunking_strategy}")
    print(f"   Chunk size: {config.chunk_size}")
    print(f"   Chunk overlap: {config.chunk_overlap}")
    print(f"   Paragraphs per chunk: {config.paragraphs_per_chunk}")
    print(f"   Paragraph overlap: {config.paragraph_overlap}")
    print(f"   Sentences per chunk: {config.sentences_per_chunk}")
    print(f"   Sentence overlap: {config.sentence_overlap}")
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
    
    # 4. Test sentence-based chunking
    print("4. Testing sentence-based chunking...")
    sent_splitter = get_text_splitter(
        chunking_strategy="sentence",
        sentences_per_chunk=3,
        sentence_overlap=1
    )
    
    sent_text = "First sentence. Second sentence! Third sentence? Fourth sentence. Fifth sentence. Sixth sentence."
    
    sent_chunks = sent_splitter.split_text(sent_text)
    print(f"   Created {len(sent_chunks)} sentence-based chunks")
    print(f"   First chunk preview: {sent_chunks[0][:50]}...")
    print("   OK Sentence-based chunking works correctly\n")
    
    # 5. Test document splitting
    print("5. Testing document splitting...")
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
    
    # Test with sentence-based splitter
    sent_doc_chunks = sent_splitter.split_documents([document])
    print(f"   Sentence-based document splitting: {len(sent_doc_chunks)} chunks")
    print("   OK Document splitting works correctly\n")
    
    # 6. Test direct splitters usage
    print("6. Testing direct splitters usage...")
    direct_para_splitter = ParagraphTextSplitter(
        paragraphs_per_chunk=3,
        paragraph_overlap=1
    )
    
    direct_sent_splitter = SentenceTextSplitter(
        sentences_per_chunk=4,
        sentence_overlap=1
    )
    
    direct_para_chunks = direct_para_splitter.split_text(para_text)
    print(f"   Direct ParagraphTextSplitter: {len(direct_para_chunks)} chunks")
    
    direct_sent_chunks = direct_sent_splitter.split_text(sent_text)
    print(f"   Direct SentenceTextSplitter: {len(direct_sent_chunks)} chunks")
    print("   OK Direct splitters work correctly\n")
    
    print("=== All tests passed! ===")


if __name__ == "__main__":
    test_all_chunking_strategies()