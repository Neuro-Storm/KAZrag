"""Module for converting TXT files to Markdown format."""

import logging
from pathlib import Path
from typing import List

from .base import BaseConverter

logger = logging.getLogger(__name__)


class TxtConverter(BaseConverter):
    """Converter for TXT files to Markdown format."""
    
    def convert(self, file_path: Path, output_dir: Path) -> List[Path]:
        """
        Convert a TXT file to Markdown format.
        
        Args:
            file_path (Path): Path to the TXT file
            output_dir (Path): Directory to save the output files
            
        Returns:
            List[Path]: List of paths to the converted Markdown files
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read TXT content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # For TXT files, we just save as MD since it's already plain text
        output_file = output_dir / f"{file_path.stem}.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
            
        logger.info(f"Converted {file_path.name} to {output_file}")
        return [output_file]
    
    def supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List[str]: List of supported extensions
        """
        return ['.txt']