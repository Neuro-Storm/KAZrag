"""Module for chunking text by paragraphs."""

import logging
from typing import List

from langchain_text_splitters.base import TextSplitter

logger = logging.getLogger(__name__)


class ParagraphTextSplitter(TextSplitter):
    """Text splitter for chunking by paragraphs."""
    
    def __init__(
        self,
        paragraphs_per_chunk: int = 3,
        paragraph_overlap: int = 1,
        keep_separator: bool = True,
        **kwargs
    ):
        """
        Initialize the paragraph text splitter.
        
        Args:
            paragraphs_per_chunk (int): Number of paragraphs per chunk
            paragraph_overlap (int): Number of paragraphs overlap between chunks
            keep_separator (bool): Whether to keep separators (newlines) in chunks
            **kwargs: Additional arguments for TextSplitter
        """
        super().__init__(**kwargs)
        self.paragraphs_per_chunk = paragraphs_per_chunk
        self.paragraph_overlap = paragraph_overlap
        self.keep_separator = keep_separator
        
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks by paragraphs.
        
        Args:
            text (str): Input text to split
            
        Returns:
            List[str]: List of text chunks
        """
        # Split text into paragraphs by double newlines
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return []
            
        chunks = []
        
        # Calculate window step
        step = self.paragraphs_per_chunk - self.paragraph_overlap
        # If step <= 0, set it to 1 to avoid infinite loop
        if step <= 0:
            step = 1
            
        i = 0
        while i < len(paragraphs):
            # Determine start and end of current chunk
            start_idx = i
            end_idx = min(i + self.paragraphs_per_chunk, len(paragraphs))
            
            # Form chunk from paragraphs
            if self.keep_separator:
                chunk = '\n\n'.join(paragraphs[start_idx:end_idx])
            else:
                chunk = ' '.join(paragraphs[start_idx:end_idx])
                
            chunks.append(chunk)
            
            # If we've reached the end of text, exit loop
            if end_idx >= len(paragraphs):
                break
                
            # Move to next chunk with overlap
            i += step
            
        return chunks


# Test function
def test_paragraph_chunking():
    """Test paragraph chunking functionality."""
    text = """First paragraph of text.

Second paragraph of text.

Third paragraph of text.

Fourth paragraph of text.

Fifth paragraph of text."""
    
    print("Source text:")
    print(repr(text))
    
    # Test with default settings (3 paragraphs per chunk, 1 paragraph overlap)
    splitter = ParagraphTextSplitter()
    print(f"\nParameters: paragraphs_per_chunk={splitter.paragraphs_per_chunk}, paragraph_overlap={splitter.paragraph_overlap}")
    
    chunks = splitter.split_text(text)
    print(f"\nChunks ({len(chunks)} pcs.):")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(repr(chunk))


if __name__ == "__main__":
    test_paragraph_chunking()