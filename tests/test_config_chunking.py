"""Test configuration with paragraph chunking strategy."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langchain_core.documents import Document

from config.settings import load_config
from core.indexing.text_splitter import TextSplitter


def test_config_paragraph_chunking():
    """Test configuration with paragraph chunking strategy."""
    # Load configuration
    config = load_config()
    
    print("=== Configuration Test ===")
    print(f"Chunking strategy: {config.chunking_strategy}")
    print(f"Paragraphs per chunk: {config.paragraphs_per_chunk}")
    print(f"Paragraph overlap: {config.paragraph_overlap}")
    
    # Create text splitter with configuration
    text_splitter = TextSplitter(config)
    
    # Sample text for testing
    sample_text = """This is the first paragraph of our test document. 
It contains some sample text to demonstrate paragraph-based chunking.

This is the second paragraph of our test document. 
It also contains sample text for demonstration purposes.

This is the third paragraph of our test document. 
We are showing how the chunker works with multiple paragraphs.

This is the fourth paragraph of our test document. 
It continues our demonstration of paragraph-based chunking.

This is the fifth paragraph of our test document. 
This is the final paragraph in our sample text."""

    document = Document(page_content=sample_text, metadata={"source": "test.txt"})
    
    # Split document
    chunks = text_splitter.split_documents([document])
    
    print(f"\nCreated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"Content: {chunk.page_content}")
        print(f"Metadata: {chunk.metadata}")


if __name__ == "__main__":
    test_config_paragraph_chunking()