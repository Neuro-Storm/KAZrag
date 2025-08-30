"""Converter using unstructured library for various document formats."""

import logging
from pathlib import Path
from typing import List

from unstructured.partition.auto import partition

from .base import BaseConverter

logger = logging.getLogger(__name__)


class UnstructuredConverter(BaseConverter):
    """Converter using unstructured library for various document formats."""
    
    def __init__(self):
        """Initialize the unstructured converter."""
        super().__init__()
        # Supported extensions by unstructured
        self._supported_extensions = {
            '.txt', '.md', '.html', '.htm', '.pdf', '.doc', '.docx', 
            '.ppt', '.pptx', '.xls', '.xlsx', '.csv', '.rtf', '.epub'
        }
    
    def supported_extensions(self) -> List[str]:
        """Get list of supported file extensions.
        
        Returns:
            List[str]: List of supported extensions
        """
        return list(self._supported_extensions)
    
    def convert(self, file_path: Path, output_dir: Path) -> List[Path]:
        """Convert a file to Markdown format using unstructured.
        
        Args:
            file_path (Path): Path to the input file
            output_dir (Path): Directory to save the output files
            
        Returns:
            List[Path]: List of paths to the converted Markdown files
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use unstructured to partition the document
            elements = partition(str(file_path))
            
            # Convert elements to markdown
            markdown_content = "\n\n".join([str(element) for element in elements])
            
            # Create output file path
            output_file = output_dir / f"{file_path.stem}.md"
            
            # Write markdown content to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Converted {file_path} to {output_file} using unstructured")
            return [output_file]
            
        except Exception as e:
            logger.error(f"Error converting {file_path} with unstructured: {e}")
            raise