"""Module for converting various file formats to Markdown."""

import logging
from pathlib import Path
from typing import List, Tuple
from config.config_manager import ConfigManager

# Import the PDF converter
from .pdf_to_md_chunker import process_pdfs_and_chunk
# Import the new converter manager
from .converters.manager import ConverterManager
# Import the new file processor
from .file_processor import FileProcessor, FileType

logger = logging.getLogger(__name__)

# Get singleton instance of ConfigManager
config_manager = ConfigManager.get_instance()

# Lazy initialization of converter manager
_converter_manager = None


def _get_converter_manager():
    """Get or create converter manager instance."""
    global _converter_manager
    if _converter_manager is None:
        _converter_manager = ConverterManager()
    return _converter_manager


def convert_files_to_md(input_dir: str, output_dir: str) -> Tuple[bool, str]:
    """
    Convert files of various formats to Markdown using the new FileProcessor.
    
    Args:
        input_dir (str): Directory containing files to convert
        output_dir (str): Directory to save converted Markdown files
        
    Returns:
        Tuple[bool, str]: (success, status message)
    """
    try:
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists() or not input_path.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
            
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize file processor
        file_processor = FileProcessor()
        
        # Scan directory for files
        files_by_type = file_processor.scan_directory(input_path, recursive=True)
        
        # Get statistics
        stats = file_processor.get_statistics(files_by_type)
        logger.info(f"Found files: {stats}")
        
        # Process files
        results = file_processor.process_files(files_by_type, output_path, _get_converter_manager())
        
        # Count successful conversions
        successful_conversions = sum(1 for result in results if result.success)
        failed_conversions = len(results) - successful_conversions
        
        if failed_conversions > 0:
            logger.warning(f"{failed_conversions} files failed to convert")
            
        logger.info(f"Successfully converted {successful_conversions} files to Markdown")
        return True, f"converted_{successful_conversions}_files"
        
    except Exception as e:
        logger.exception(f"Error converting files: {e}")
        return False, f"conversion_error: {str(e)}"


def get_supported_formats() -> List[str]:
    """
    Get list of supported file formats.
    
    Returns:
        List[str]: List of supported formats
    """
    # Start with formats handled by MinerU
    formats = ['.pdf', '.djvu', '.jpg', '.jpeg', '.png']
    
    # Add formats supported by our converters
    formats.extend(_get_converter_manager().get_supported_extensions())
    
    # Remove duplicates and return
    return list(set(formats))