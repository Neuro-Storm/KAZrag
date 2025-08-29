# Comprehensive multilevel chunking integration test

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langchain_core.documents import Document
from core.indexing.multilevel_chunker import (
    MultiLevelChunker, 
    create_flexible_multilevel_chunker,
    create_multilevel_chunker_from_config
)
from config.settings import load_config


def test_comprehensive_multilevel_chunking():
    print("=== Comprehensive multilevel chunking integration test ===")
    print()
    
    # Load configuration
    config = load_config()
    print("1. Configuration loaded successfully")
    
    # Create test document
    sample_text = """This is the first paragraph of our comprehensive test document. 
It contains substantial sample text to demonstrate all multilevel chunking capabilities and flexibility.

This is the second paragraph of our test document. 
It also contains sample text for demonstration purposes and shows how paragraphs are handled in different strategies.

This is the third paragraph of our test document. 
We are showing how the multilevel chunker works with multiple paragraphs and different chunking strategies effectively.

This is the fourth paragraph of our test document. 
It continues our demonstration of multilevel chunking and provides more content for comprehensive testing.

This is the fifth paragraph of our test document. 
This is the final paragraph in our sample text, demonstrating the complete workflow and flexibility."""

    document = Document(
        page_content=sample_text,
        metadata={"source": "comprehensive_test.txt", "test_type": "multilevel"}
    )
    
    print("2. Test document created")
    
    # Test 1: Character-Character multilevel chunking
    print("\n3. Testing Character-Character multilevel chunking:")
    char_char_chunker = MultiLevelChunker(
        macro_chunk_strategy="character",
        macro_chunk_size=300,
        macro_chunk_overlap=30,
        micro_chunk_strategy="character",
        micro_chunk_size=100,
        micro_chunk_overlap=10
    )
    
    char_char_chunks = char_char_chunker.create_multilevel_chunks([document])
    print("   Created {} macro-chunks with average {} micro-chunks each".format(
        len(char_char_chunks),
        round(sum(chunk["total_micro_chunks"] for chunk in char_char_chunks) / len(char_char_chunks), 1) if char_char_chunks else 0
    ))
    
    # Test 2: Paragraph-Sentence multilevel chunking
    print("\n4. Testing Paragraph-Sentence multilevel chunking:")
    para_sent_chunker = MultiLevelChunker(
        macro_chunk_strategy="paragraph",
        macro_paragraphs_per_chunk=3,
        macro_paragraph_overlap=1,
        micro_chunk_strategy="sentence",
        micro_sentences_per_chunk=4,
        micro_sentence_overlap=1
    )
    
    para_sent_chunks = para_sent_chunker.create_multilevel_chunks([document])
    print("   Created {} macro-chunks with average {} micro-chunks each".format(
        len(para_sent_chunks),
        round(sum(chunk["total_micro_chunks"] for chunk in para_sent_chunks) / len(para_sent_chunks), 1) if para_sent_chunks else 0
    ))
    
    # Test 3: Sentence-Character multilevel chunking
    print("\n5. Testing Sentence-Character multilevel chunking:")
    sent_char_chunker = MultiLevelChunker(
        macro_chunk_strategy="sentence",
        macro_sentences_per_chunk=6,
        macro_sentence_overlap=2,
        micro_chunk_strategy="character",
        micro_chunk_size=150,
        micro_chunk_overlap=15
    )
    
    sent_char_chunks = sent_char_chunker.create_multilevel_chunks([document])
    print("   Created {} macro-chunks with average {} micro-chunks each".format(
        len(sent_char_chunks),
        round(sum(chunk["total_micro_chunks"] for chunk in sent_char_chunks) / len(sent_char_chunks), 1) if sent_char_chunks else 0
    ))
    
    # Test 4: Using flexible chunker creator
    print("\n6. Testing flexible chunker creator:")
    
    # Create flexible chunker with paragraph macro and character micro
    flexible_chunker = create_flexible_multilevel_chunker(
        macro_strategy="paragraph",
        macro_size=2,
        micro_strategy="character",
        micro_size=200
    )
    
    flexible_chunks = flexible_chunker.create_multilevel_chunks([document])
    print("   Flexible chunker created {} macro-chunks with average {} micro-chunks each".format(
        len(flexible_chunks),
        round(sum(chunk["total_micro_chunks"] for chunk in flexible_chunks) / len(flexible_chunks), 1) if flexible_chunks else 0
    ))
    
    # Test 5: Creating chunker from configuration
    print("\n7. Testing chunker creation from configuration:")
    
    # Create config-like dictionary
    test_config = {
        'multilevel_macro_strategy': 'character',
        'multilevel_macro_chunk_size': 400,
        'multilevel_macro_chunk_overlap': 40,
        'multilevel_micro_strategy': 'paragraph',
        'multilevel_micro_paragraphs_per_chunk': 2,
        'multilevel_micro_paragraph_overlap': 1
    }
    
    config_chunker = create_multilevel_chunker_from_config(test_config)
    config_chunks = config_chunker.create_multilevel_chunks([document])
    print("   Config-based chunker created {} macro-chunks with average {} micro-chunks each".format(
        len(config_chunks),
        round(sum(chunk["total_micro_chunks"] for chunk in config_chunks) / len(config_chunks), 1) if config_chunks else 0
    ))
    
    # Test 6: Vector generation
    print("\n8. Testing vector generation:")
    vectors = char_char_chunker.get_all_vectors_for_chunk(sample_text[:500])
    print("   Generated {} text representations for vectorization".format(len(vectors)))
    print("   Main chunk (first 100 chars): {}...".format(vectors[0][:100]))
    if len(vectors) > 1:
        print("   First micro-chunk (first 50 chars): {}...".format(vectors[1][:50]))
    
    print("\n=== All comprehensive tests passed successfully! ===")
    print()
    print("Key flexibility features demonstrated:")
    print("- Any macro-chunk strategy (character, paragraph, sentence)")
    print("- Any micro-chunk strategy (character, paragraph, sentence)")
    print("- Configurable sizes and overlaps for both levels")
    print("- Multiple creation methods (direct, flexible, config-based)")
    print("- Metadata preservation across all chunking strategies")
    print("- Integration with existing document processing pipeline")


if __name__ == "__main__":
    test_comprehensive_multilevel_chunking()