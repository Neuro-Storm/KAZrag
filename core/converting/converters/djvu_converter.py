"""Module for converting DJVU files to PDF format."""

import logging
import subprocess
from pathlib import Path
from typing import List

from .base import BaseConverter

logger = logging.getLogger(__name__)


class DjvuConverter(BaseConverter):
    """Converter for DJVU files to PDF format (intermediate step for Markdown conversion)."""
    
    def convert(self, file_path: Path, output_dir: Path) -> List[Path]:
        """
        Convert a DJVU file to PDF format.
        
        Args:
            file_path (Path): Path to the DJVU file
            output_dir (Path): Directory to save the output files
            
        Returns:
            List[Path]: List of paths to the converted PDF files
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Check if ddjvu tool is available
        try:
            subprocess.run(["ddjvu", "--version"], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL, 
                         check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise ImportError("ddjvu tool is required for DJVU conversion. Please install DjVuLibre.")
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert DJVU to PDF
        output_file = output_dir / f"{file_path.stem}.pdf"
        
        try:
            # Use ddjvu to convert DJVU to PDF
            cmd = [
                "ddjvu",
                "-format=pdf",
                str(file_path),
                str(output_file)
            ]
            
            subprocess.run(cmd, 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE, 
                          check=True)
            
            logger.info(f"Converted {file_path.name} to {output_file}")
            return [output_file]
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error converting DJVU to PDF: {e.stderr}")
    
    def supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List[str]: List of supported extensions
        """
        return ['.djvu']