"""Integration tests for the indexing process."""

from unittest.mock import patch

import pytest

from config.settings import Config
from core.file_converter import (
    run_multi_format_processing_from_config,
    run_pdf_processing_from_config,
)
from core.indexer import run_indexing_logic


class TestIndexingProcess:
    """Integration tests for the indexing process."""

    @pytest.mark.asyncio
    async def test_indexing_process_folder_not_found(self):
        """Test the indexing process when folder is not found."""
        # Create mock config
        config = Config()
        config.folder_path = "./nonexistent_folder"  # Папка не существует
        config.collection_name = "test-collection"
        config.current_hf_model = "sentence-transformers/all-MiniLM-L6-v2"
        config.chunk_size = 100
        config.chunk_overlap = 20
        config.device = "cpu"
        config.index_dense = True
        config.index_bm25 = False
        config.index_hybrid = False
        config.embedding_batch_size = 32
        config.indexing_batch_size = 50
        
        with patch('core.indexer.config_manager') as mock_config_manager:
            mock_config_manager.get_instance().get.return_value = config
            
            success, status = await run_indexing_logic()
            
            # Функция должна вернуть False и "folder_not_found" для несуществующей папки
            assert success is False
            assert status == "folder_not_found"

    @pytest.mark.asyncio
    async def test_indexing_process_no_index_type(self):
        """Test indexing process when no index type is configured."""
        # Create mock config with no index type but existing folder path
        config = Config()
        config.folder_path = "./test_folder"  # Путь к несуществующей папке
        config.collection_name = "test-collection"
        config.index_dense = False
        config.index_bm25 = False
        config.index_hybrid = False
        
        with patch('core.indexer.config_manager') as mock_config_manager:
            mock_config_manager.get_instance().get.return_value = config
            
            success, status = await run_indexing_logic()
            
            # Должна вернуться ошибка no_index_type, потому что проверка типа индексации идет первой
            assert success is False
            assert status == "no_index_type"

    def test_pdf_processing_integration(self):
        """Test PDF processing integration."""
        # Create mock config
        config = Config()
        config.mineru_input_pdf_dir = "./pdfs_to_process"
        config.mineru_output_md_dir = "./data_to_index"
        config.mineru_enable_formula_parsing = False
        config.mineru_enable_table_parsing = False
        config.mineru_model_source = "huggingface"
        config.mineru_backend = "pipeline"
        config.mineru_method = "auto"
        config.mineru_lang = "east_slavic"
        config.device = "cpu"
        
        with patch('core.file_converter.config_manager') as mock_config_manager:
            mock_config_manager.get_instance().get.return_value = config
            
            with patch('core.file_converter.process_pdfs_and_chunk') as mock_process_pdfs:
                mock_process_pdfs.return_value = (True, "pdfs_processed_successfully")
                
                success, status = run_pdf_processing_from_config()
                
                assert success is True
                assert status == "pdfs_processed_successfully"

    def test_multi_format_processing_integration(self):
        """Test multi-format file processing integration."""
        # Create mock config
        config = Config()
        config.mineru_input_pdf_dir = "./pdfs_to_process"
        config.mineru_output_md_dir = "./data_to_index"
        
        with patch('core.file_converter.config_manager') as mock_config_manager:
            mock_config_manager.get_instance().get.return_value = config
            
            with patch('core.file_converter.convert_files_to_md') as mock_convert_files:
                mock_convert_files.return_value = (True, "converted_5_files")
                
                success, status = run_multi_format_processing_from_config()
                
                assert success is True
                assert "files_processed_successfully" in status


class TestIndexingErrorHandling:
    """Tests for error handling in indexing process."""

    @pytest.mark.asyncio
    async def test_indexing_process_indexing_error(self):
        """Test indexing process when indexing fails."""
        # Create mock config
        config = Config()
        config.folder_path = "./test_folder"  # Несуществующая папка
        config.collection_name = "test-collection"
        config.index_dense = True
        config.index_bm25 = False
        config.index_hybrid = False
        config.current_hf_model = "sentence-transformers/all-MiniLM-L6-v2"
        config.chunk_size = 100
        config.chunk_overlap = 20
        config.device = "cpu"
        config.embedding_batch_size = 32
        config.indexing_batch_size = 50
        
        with patch('core.indexer.config_manager') as mock_config_manager:
            mock_config_manager.get_instance().get.return_value = config
            
            # Поскольку папка не существует, функция должна вернуть "folder_not_found"
            success, status = await run_indexing_logic()
            assert success is False
            assert status == "folder_not_found"

    @pytest.mark.asyncio
    async def test_indexing_process_file_loading_error(self):
        """Test indexing process when file loading fails."""
        # Create mock config
        config = Config()
        config.folder_path = "./test_folder"  # Несуществующая папка
        config.collection_name = "test-collection"
        config.index_dense = True
        config.index_bm25 = False
        config.index_hybrid = False
        config.current_hf_model = "sentence-transformers/all-MiniLM-L6-v2"
        config.chunk_size = 100
        config.chunk_overlap = 20
        config.device = "cpu"
        config.embedding_batch_size = 32
        config.indexing_batch_size = 50
        
        with patch('core.indexer.config_manager') as mock_config_manager:
            mock_config_manager.get_instance().get.return_value = config
            
            # Поскольку папка не существует, функция должна вернуть "folder_not_found"
            success, status = await run_indexing_logic()
            assert success is False
            assert status == "folder_not_found"