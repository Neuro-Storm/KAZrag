"""Module for converting image files to Markdown format using MinerU."""

import logging
from pathlib import Path
from typing import List

from .base import BaseConverter

logger = logging.getLogger(__name__)


class ImageConverter(BaseConverter):
    """Converter for image files to Markdown format (using MinerU)."""
    
    def convert(self, file_path: Path, output_dir: Path) -> List[Path]:
        """
        Images can be processed directly by MinerU, so we just return the file path.
        The actual conversion will happen in the multi-format converter.
        
        Args:
            file_path (Path): Path to the image file
            output_dir (Path): Directory to save the output files
            
        Returns:
            List[Path]: List of paths to the image files (will be processed by MinerU)
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        logger.info(f"Image file {file_path.name} will be processed by MinerU")
        return [file_path]
    
    def supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List[str]: List of supported extensions
        """
        return ['.jpg', '.jpeg', '.png']