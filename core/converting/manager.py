"""Module for managing format converters."""

import logging
from pathlib import Path
from typing import List, Dict
import importlib

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
        converter_modules = [
            ('docx', '.converting.converters.docx_converter'),
            ('txt', '.converting.converters.txt_converter'),
            ('html', '.converting.converters.html_converter'),
            ('djvu', '.converting.converters.djvu_converter'),
            ('image', '.converting.converters.image_converter'),
            ('presentation', '.converting.converters.presentation_converter'),
            ('excel', '.converting.converters.excel_converter'),
        ]
        
        for name, module_path in converter_modules:
            try:
                # Import the module
                module = importlib.import_module(module_path, package='core.converting.converters')
                converter_class = getattr(module, f'{name.capitalize()}Converter')
                converter = converter_class()
                self.converters[name] = converter
                
                # Register supported extensions
                for ext in converter.supported_extensions():
                    if ext not in self.converters:
                        self.converters[ext] = converter
                    else:
                        logger.warning(f"Extension {ext} already registered, skipping {name} converter")
                        
                logger.info(f"Loaded {name} converter")
            except ImportError as e:
                logger.warning(f"Could not load {name} converter: {e}")
            except Exception as e:
                logger.error(f"Error loading {name} converter: {e}")
    
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
        
        # Get file extension
        ext = file_path.suffix.lower()
        
        # Check if we have a converter for this extension
        if ext not in self.converters:
            raise ValueError(f"No converter available for extension: {ext}")
            
        # Use the appropriate converter
        converter = self.converters[ext]
        try:
            return converter.convert(file_path, output_dir)
        except Exception as e:
            logger.error(f"Error converting {file_path.name} with {converter.__class__.__name__}: {e}")
            raise  # Re-raise the exception to be handled by the caller
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of all supported file extensions.
        
        Returns:
            List[str]: List of supported extensions
        """
        # Initialize if needed
        self._initialize_if_needed()
        
        extensions = set()
        for converter in self.converters.values():
            if hasattr(converter, 'supported_extensions'):
                extensions.update(converter.supported_extensions())
        return list(extensions)