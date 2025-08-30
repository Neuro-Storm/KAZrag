"""Module for managing format converters."""

import importlib
import logging
from pathlib import Path
from typing import Dict, List

from .converters.base import BaseConverter

logger = logging.getLogger(__name__)


class ConverterManager:
    """Manager for handling different format converters."""
    
    def __init__(self):
        """Initialize the converter manager with available converters."""
        self.converters: Dict[str, BaseConverter] = {}
        self._initialized = False
    
    def _initialize_if_needed(self):
        """Initialize converters only when needed."""
        if not self._initialized:
            self._load_converters()
            self._initialized = True
    
    def _load_converters(self):
        """Load available converters dynamically."""
        # Use unstructured converter for all supported formats
        try:
            from .converters.unstructured_converter import UnstructuredConverter
            converter = UnstructuredConverter()
            self.converters['unstructured'] = converter
            logger.info("Loaded unstructured converter for all supported formats")
        except ImportError as e:
            logger.warning(f"Could not load unstructured converter: {e}")
        except Exception as e:
            logger.error(f"Error loading unstructured converter: {e}")
    
    def convert_file(self, file_path: Path, output_dir: Path) -> List[Path]:
        """
        Convert a file to Markdown format using the appropriate converter.
        
        Args:
            file_path (Path): Path to the input file
            output_dir (Path): Directory to save the output files
            
        Returns:
            List[Path]: List of paths to the converted Markdown files
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Initialize if needed
        self._initialize_if_needed()
        
        # Use unstructured converter for all files
        if 'unstructured' in self.converters:
            converter = self.converters['unstructured']
            try:
                return converter.convert(file_path, output_dir)
            except Exception as e:
                logger.error(f"Error converting {file_path.name} with {converter.__class__.__name__}: {e}")
                raise  # Re-raise the exception to be handled by the caller
        else:
            raise ValueError(f"No converter available for file: {file_path}")
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of all supported file extensions.
        
        Returns:
            List[str]: List of supported extensions
        """
        # Initialize if needed
        self._initialize_if_needed()
        
        # Get supported extensions from unstructured converter
        if 'unstructured' in self.converters:
            converter = self.converters['unstructured']
            if hasattr(converter, 'supported_extensions'):
                return converter.supported_extensions()
        
        return []