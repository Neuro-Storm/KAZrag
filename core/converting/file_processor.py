"""Module for processing files and directories with advanced features."""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


class FileType(Enum):
    """Enumeration of supported file types."""
    PDF = "pdf"
    DJVU = "djvu"
    IMAGE = "image"
    DOCUMENT = "document"
    PRESENTATION = "presentation"
    SPREADSHEET = "spreadsheet"
    HTML = "html"
    UNKNOWN = "unknown"


@dataclass
class FileProcessingResult:
    """Data class for file processing results."""
    file_path: Path
    output_path: Optional[Path]
    success: bool
    error_message: Optional[str]
    processing_time: float = 0.0
    metadata: Dict = None


class FileProcessor:
    """Advanced file processor with scanning, grouping, and conversion capabilities."""
    
    # Supported file extensions by type
    FILE_EXTENSIONS = {
        FileType.PDF: {'.pdf'},
        FileType.DJVU: {'.djvu'},
        FileType.IMAGE: {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'},
        FileType.DOCUMENT: {'.docx', '.doc', '.txt', '.md', '.rtf'},
        FileType.PRESENTATION: {'.pptx', '.ppt'},
        FileType.SPREADSHEET: {'.xlsx', '.xls', '.csv'},
        FileType.HTML: {'.html', '.htm'}
    }
    
    def __init__(self, progress_callback: Optional[Callable[[int, int], None]] = None):
        """Initialize FileProcessor.
        
        Args:
            progress_callback: Optional callback function for progress reporting
        """
        self.progress_callback = progress_callback
        self.supported_extensions = self._get_all_supported_extensions()
        
    def _get_all_supported_extensions(self) -> Set[str]:
        """Get all supported file extensions."""
        extensions = set()
        for ext_set in self.FILE_EXTENSIONS.values():
            extensions.update(ext_set)
        return extensions
    
    def scan_directory(
        self, 
        directory: Path, 
        recursive: bool = True
    ) -> Dict[FileType, List[Path]]:
        """Scan directory and identify files by type using pandas DataFrame.
        
        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories recursively
            
        Returns:
            Dict[FileType, List[Path]]: Files grouped by type
            
        Raises:
            FileNotFoundError: If directory doesn't exist
            PermissionError: If no permission to access directory
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
            
        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")
            
        try:
            # Use pathlib.Path.rglob for file scanning
            if recursive:
                all_files = list(directory.rglob("*"))
            else:
                all_files = list(directory.iterdir())
                
            # Filter only files (not directories)
            files = [f for f in all_files if f.is_file()]
            
            # Create DataFrame for efficient processing
            if files:
                df = pd.DataFrame({
                    'path': files,
                    'name': [f.name for f in files],
                    'extension': [f.suffix.lower() for f in files],
                    'size': [f.stat().st_size for f in files]
                })
                
                # Identify file types
                def identify_file_type(ext):
                    for file_type, extensions in self.FILE_EXTENSIONS.items():
                        if ext in extensions:
                            return file_type
                    return FileType.UNKNOWN
                
                df['file_type'] = df['extension'].apply(identify_file_type)
                
                # Group files by type
                files_by_type = {}
                for file_type in FileType:
                    file_paths = df[df['file_type'] == file_type]['path'].tolist()
                    files_by_type[file_type] = file_paths
            else:
                # Initialize empty lists for all file types
                files_by_type = {file_type: [] for file_type in FileType}
                
            # Log results
            total_files = sum(len(files) for files in files_by_type.values())
            logger.info(f"Scanned {total_files} files in {directory}")
            
            return files_by_type
            
        except PermissionError:
            raise PermissionError(f"No permission to access directory: {directory}")
        except Exception as e:
                logger.exception(f"Error scanning directory {directory}: {e}")
                raise
    
    def process_files(
        self, 
        files_by_type: Dict[FileType, List[Path]], 
        output_dir: Path,
        converter_manager=None  # Will be imported when needed
    ) -> List[FileProcessingResult]:
        """Process files of different types using appropriate converters.
        
        Args:
            files_by_type: Files grouped by type
            output_dir: Output directory for processed files
            converter_manager: Converter manager instance (optional)
            
        Returns:
            List[FileProcessingResult]: Processing results
        """
        results = []
        total_files = sum(len(files) for files in files_by_type.values())
        processed_count = 0
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import converter manager if not provided
        if converter_manager is None:
            from core.converting.manager import ConverterManager
            converter_manager = ConverterManager()
        
        # Process each file type
        for file_type, files in files_by_type.items():
            if not files:
                continue
                
            logger.info(f"Processing {len(files)} {file_type.value} files")
            
            for file_path in files:
                try:
                    # Report progress
                    processed_count += 1
                    if self.progress_callback:
                        self.progress_callback(processed_count, total_files)
                    
                    # Process file based on type
                    output_path = self._process_file_by_type(
                        file_path, 
                        file_type, 
                        output_dir, 
                        converter_manager
                    )
                    
                    # Check if processing was successful
                    if output_path is not None:
                        results.append(FileProcessingResult(
                            file_path=file_path,
                            output_path=output_path,
                            success=True,
                            error_message=None,
                            metadata={"file_type": file_type.value}
                        ))
                    else:
                        results.append(FileProcessingResult(
                            file_path=file_path,
                            output_path=None,
                            success=False,
                            error_message="File processing failed (no output path returned)",
                            metadata={"file_type": file_type.value}
                        ))
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    results.append(FileProcessingResult(
                        file_path=file_path,
                        output_path=None,
                        success=False,
                        error_message=str(e),
                        metadata={"file_type": file_type.value}
                    ))
        
        return results
    
    def _process_file_by_type(
        self, 
        file_path: Path, 
        file_type: FileType, 
        output_dir: Path,
        converter_manager
    ) -> Optional[Path]:
        """Process file based on its type.
        
        Args:
            file_path: Path to file
            file_type: File type
            output_dir: Output directory
            converter_manager: Converter manager instance
            
        Returns:
            Optional[Path]: Output file path or None if failed
        """
        try:
            # For PDF, DJVU, and image files, we might use MinerU or other specialized tools
            if file_type in [FileType.PDF, FileType.DJVU, FileType.IMAGE]:
                # For now, delegate to existing conversion logic
                # In the future, we can implement more sophisticated processing
                return self._process_with_mineru(file_path, file_type, output_dir)
            else:
                # Use converter manager for other file types
                converted_files = converter_manager.convert_file(file_path, output_dir)
                return converted_files[0] if converted_files else None
                
        except Exception as e:
            logger.error(f"Error processing {file_path} as {file_type.value}: {e}")
            raise
    
    def _process_with_mineru(
        self, 
        file_path: Path, 
        file_type: FileType, 
        output_dir: Path
    ) -> Optional[Path]:
        """Process file with MinerU service.
        
        Args:
            file_path: Path to file
            file_type: File type
            output_dir: Output directory
            
        Returns:
            Optional[Path]: Output file path or None if failed
        """
        try:
            # Import the MinerU service
            from core.converting.mineru_service import MinerUService
            
            # Create MinerU service instance
            mineru_service = MinerUService()
            
            # Check if MinerU is available
            if not mineru_service.is_mineru_available():
                logger.error("MinerU is not available. Please install the 'mineru' package.")
                return None
            
            # Get configuration
            from config.config_manager import ConfigManager
            config_manager = ConfigManager.get_instance()
            config = config_manager.get()
            
            # Create temporary input directory for this file
            temp_input_dir = output_dir / "temp_input"
            temp_input_dir.mkdir(exist_ok=True)
            
            # Copy file to temporary input directory
            temp_file_path = temp_input_dir / file_path.name
            with open(file_path, 'rb') as src, open(temp_file_path, 'wb') as dst:
                dst.write(src.read())
            
            # Process with MinerU service
            success = mineru_service.process_pdfs(
                input_pdf_dir=str(temp_input_dir),
                output_md_dir=str(output_dir),
                enable_formula_parsing=config.mineru_enable_formula_parsing,
                enable_table_parsing=config.mineru_enable_table_parsing,
                model_source=config.mineru_model_source,
                models_dir=config.mineru_models_dir if config.mineru_models_dir else None,
                backend=config.mineru_backend,
                method=config.mineru_method,
                lang=config.mineru_lang,
                sglang_url=config.mineru_sglang_url if config.mineru_sglang_url else None
            )
            
            # Clean up temporary input directory
            import shutil
            shutil.rmtree(temp_input_dir, ignore_errors=True)
            
            if success:
                # Return the output file path
                output_file = output_dir / f"{file_path.stem}.md"
                return output_file if output_file.exists() else None
            else:
                return None
            
        except Exception as e:
            logger.error(f"Error processing {file_path} with MinerU service: {e}")
            # Clean up temporary input directory if it exists
            temp_input_dir = output_dir / "temp_input"
            if temp_input_dir.exists():
                import shutil
                shutil.rmtree(temp_input_dir, ignore_errors=True)
            return None
    
    def get_statistics(self, files_by_type: Dict[FileType, List[Path]]) -> Dict[str, int]:
        """Get statistics about files by type using pandas DataFrame.
        
        Args:
            files_by_type: Files grouped by type
            
        Returns:
            Dict[str, int]: Statistics
        """
        # Create DataFrame for efficient statistics calculation
        data = []
        total = 0
        
        for file_type, files in files_by_type.items():
            count = len(files)
            data.append({'file_type': file_type.value, 'count': count})
            total += count
            
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Convert to dictionary
        stats = df.set_index('file_type')['count'].to_dict()
        stats['total'] = total
        
        return stats