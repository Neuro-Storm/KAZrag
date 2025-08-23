"""Base class for format converters."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List


class BaseConverter(ABC):
    """Abstract base class for format converters."""
    
    @abstractmethod
    def convert(self, file_path: Path, output_dir: Path) -> List[Path]:
        """
        Convert a file to Markdown format.
        
        Args:
            file_path (Path): Path to the input file
            output_dir (Path): Directory to save the output files
            
        Returns:
            List[Path]: List of paths to the converted Markdown files
        """
        pass
    
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List[str]: List of supported extensions (e.g., ['.docx', '.txt'])
        """
        pass