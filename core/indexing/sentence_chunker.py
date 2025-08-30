"""Module for chunking text by sentences."""

import logging
import re
from typing import List

from langchain_text_splitters.base import TextSplitter

logger = logging.getLogger(__name__)


class SentenceTextSplitter(TextSplitter):
    """Text splitter for chunking by sentences."""
    
    def __init__(
        self,
        sentences_per_chunk: int = 5,
        sentence_overlap: int = 1,
        keep_separator: bool = True,
        **kwargs
    ):
        """
        Initialize the sentence text splitter.
        
        Args:
            sentences_per_chunk (int): Number of sentences per chunk
            sentence_overlap (int): Number of sentences overlap between chunks
            keep_separator (bool): Whether to keep separators (newlines) in chunks
            **kwargs: Additional arguments for TextSplitter
        """
        super().__init__(**kwargs)
        self.sentences_per_chunk = sentences_per_chunk
        self.sentence_overlap = sentence_overlap
        self.keep_separator = keep_separator
        
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks by sentences.
        
        Args:
            text (str): Input text to split
            
        Returns:
            List[str]: List of text chunks
        """
        # More accurate sentence splitting with punctuation preservation
        # Use regex to find sentence endings with punctuation capture
        sentence_endings = re.finditer(r'[.!?]+', text)
        sentences = []
        last_end = 0
        
        for match in sentence_endings:
            # Add sentence with punctuation preserved
            sentence = text[last_end:match.end()].strip()
            if sentence:
                sentences.append(sentence)
            last_end = match.end()
        
        # Add remaining text if any
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                sentences.append(remaining)
        
        # If no sentences found by punctuation, split by periods
        if not sentences:
            sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
            # Remove trailing period if it was added extra
            if sentences and sentences[-1].endswith('.'):
                sentences[-1] = sentences[-1][:-1]
        
        if not sentences:
            return []
            
        chunks = []
        
        # Calculate window step
        step = self.sentences_per_chunk - self.sentence_overlap
        # If step <= 0, set it to 1 to avoid infinite loop
        if step <= 0:
            step = 1
            
        i = 0
        while i < len(sentences):
            # Determine start and end of current chunk
            start_idx = i
            end_idx = min(i + self.sentences_per_chunk, len(sentences))
            
            # Form chunk from sentences
            if self.keep_separator:
                chunk = " ".join(sentences[start_idx:end_idx])
            else:
                # Remove punctuation for chunk without separators
                cleaned_sentences = [re.sub(r'[.!?]+$', '', s).strip() for s in sentences[start_idx:end_idx]]
                chunk = " ".join(cleaned_sentences)
                
            chunks.append(chunk)
            
            # If we've reached the end of text, exit loop
            if end_idx >= len(sentences):
                break
                
            # Move to next chunk with overlap
            i += step
            
        return chunks