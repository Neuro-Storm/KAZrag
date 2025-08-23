"""Module for converting various file formats to Markdown."""

import logging
from pathlib import Path
from typing import List, Tuple
from config.settings import load_config, Config

# Import the PDF converter
from .pdf_to_md_chunker import process_pdfs_and_chunk
# Import the new converter manager
from .converters.manager import ConverterManager

logger = logging.getLogger(__name__)

# Initialize the converter manager
converter_manager = ConverterManager()


def convert_files_to_md(input_dir: str, output_dir: str) -> Tuple[bool, str]:
    """
    Convert files of various formats to Markdown.
    
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
        
        # Get all files in the input directory
        files = [f for f in input_path.iterdir() if f.is_file()]
        
        if not files:
            logger.info(f"No files found in {input_dir}")
            return True, "no_files_found"
        
        # Categorize files by type
        pdf_files = [f for f in files if f.suffix.lower() == '.pdf']
        djvu_files = [f for f in files if f.suffix.lower() == '.djvu']
        image_files = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        other_files = [f for f in files if f.suffix.lower() not in ['.pdf', '.djvu', '.jpg', '.jpeg', '.png']]
        
        # Process files
        processed_count = 0
        
        # Process DJVU files (convert to PDF first)
        if djvu_files:
            logger.info(f"Processing {len(djvu_files)} DJVU files")
            for djvu_file in djvu_files:
                try:
                    # Convert DJVU to PDF
                    pdf_paths = converter_manager.convert_file(djvu_file, input_path)
                    if pdf_paths:
                        pdf_files.extend(pdf_paths)  # Add converted PDFs to PDF processing queue
                        processed_count += 1
                except Exception as e:
                    logger.error(f"Error converting DJVU {djvu_file.name}: {e}")
        
        # Process PDF files (including converted DJVU files) using MinerU
        if pdf_files:
            logger.info(f"Processing {len(pdf_files)} PDF files with MinerU")
            # Load config for PDF processing settings
            config: Config = load_config()
            
            # Process each PDF file
            for pdf_file in pdf_files:
                try:
                    pdf_output_dir = output_path / pdf_file.stem
                    pdf_output_dir.mkdir(exist_ok=True)
                    
                    process_pdfs_and_chunk(
                        input_pdf_dir=str(pdf_file.parent),
                        output_md_dir=str(pdf_output_dir),
                        enable_formula_parsing=config.mineru_enable_formula_parsing,
                        enable_table_parsing=config.mineru_enable_table_parsing,
                        model_source=config.mineru_model_source,
                        models_dir=config.mineru_models_dir if config.mineru_models_dir else None,
                        backend=config.mineru_backend,
                        method=config.mineru_method,
                        lang=config.mineru_lang,
                        sglang_url=config.mineru_sglang_url if config.mineru_sglang_url else None,
                        device=config.device
                    )
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Error processing PDF {pdf_file.name}: {e}")
        
        # Process image files using MinerU
        if image_files:
            logger.info(f"Processing {len(image_files)} image files with MinerU")
            # Load config for PDF processing settings
            config: Config = load_config()
            
            # Create a temporary directory for image processing
            temp_image_dir = input_path / "temp_images"
            temp_image_dir.mkdir(exist_ok=True)
            
            try:
                # Move image files to temp directory
                for image_file in image_files:
                    temp_image_path = temp_image_dir / image_file.name
                    image_file.rename(temp_image_path)
                
                # Process images as PDFs using MinerU
                try:
                    image_output_dir = output_path / "images_processed"
                    image_output_dir.mkdir(exist_ok=True)
                    
                    process_pdfs_and_chunk(
                        input_pdf_dir=str(temp_image_dir),
                        output_md_dir=str(image_output_dir),
                        enable_formula_parsing=config.mineru_enable_formula_parsing,
                        enable_table_parsing=config.mineru_enable_table_parsing,
                        model_source=config.mineru_model_source,
                        models_dir=config.mineru_models_dir if config.mineru_models_dir else None,
                        backend=config.mineru_backend,
                        method=config.mineru_method,
                        lang=config.mineru_lang,
                        sglang_url=config.mineru_sglang_url if config.mineru_sglang_url else None,
                        device=config.device
                    )
                    processed_count += len(image_files)
                except Exception as e:
                    logger.error(f"Error processing images: {e}")
            finally:
                # Move image files back and clean up temp directory
                for temp_image_file in temp_image_dir.iterdir():
                    if temp_image_file.is_file():
                        original_path = input_path / temp_image_file.name
                        temp_image_file.rename(original_path)
                # Remove temp directory
                if temp_image_dir.exists():
                    temp_image_dir.rmdir()
        
        # Process other files using appropriate converters
        for file_path in other_files:
            try:
                converter_manager.convert_file(file_path, output_path)
                processed_count += 1
            except Exception as e:
                logger.error(f"Error converting {file_path.name}: {e}")
        
        logger.info(f"Successfully converted {processed_count} files to Markdown")
        return True, f"converted_{processed_count}_files"
        
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
    formats.extend(converter_manager.get_supported_extensions())
    
    # Remove duplicates and return
    return list(set(formats))