"""Service for handling MinerU operations."""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class MinerUService:
    """Service for handling MinerU operations."""
    
    def __init__(self):
        """Initialize the MinerU service."""
        pass
    
    def process_pdfs(
        self,
        input_pdf_dir: str,
        output_md_dir: str,
        enable_formula_parsing: bool = False,
        enable_table_parsing: bool = False,
        model_source: str = "huggingface",
        models_dir: Optional[str] = None,
        backend: str = "pipeline",
        method: str = "auto",
        lang: str = "east_slavic",
        sglang_url: Optional[str] = None
    ) -> bool:
        """Process PDF files using MinerU.
        
        Args:
            input_pdf_dir (str): Path to directory with PDF files.
            output_md_dir (str): Path to save results (markdown + images).
            enable_formula_parsing (bool): Enable formula parsing (default: False).
            enable_table_parsing (bool): Enable table parsing (default: False).
            model_source (str): Model source: 'huggingface', 'modelscope' or 'local'.
            models_dir (Optional[str]): Path to local models (only for model_source='local').
            backend (str): Processing backend: 'pipeline', 'vlm-transformers' or 'vlm-sglang-client'.
            method (str): Parsing method: 'auto', 'txt' or 'ocr'.
            lang (str): OCR language (default: 'east_slavic' for East Slavic languages).
            sglang_url (Optional[str]): Sglang server URL (only for backend='vlm-sglang-client').
            
        Returns:
            bool: True if all files were processed successfully, False otherwise.
        """
        try:
            # Import the MinerU processing function
            from core.converting.pdf_to_md_chunker import process_pdfs_and_chunk
            
            # Process PDFs and return the result
            return process_pdfs_and_chunk(
                input_pdf_dir=input_pdf_dir,
                output_md_dir=output_md_dir,
                enable_formula_parsing=enable_formula_parsing,
                enable_table_parsing=enable_table_parsing,
                model_source=model_source,
                models_dir=models_dir,
                backend=backend,
                method=method,
                lang=lang,
                sglang_url=sglang_url
            )
            
        except Exception as e:
            logger.error(f"Error processing PDFs with MinerU: {e}")
            return False
    
    def is_mineru_available(self) -> bool:
        """Check if MinerU is available.
        
        Returns:
            bool: True if MinerU is available, False otherwise
        """
        try:
            import subprocess
            result = subprocess.run(["mineru", "--help"], 
                                  capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False