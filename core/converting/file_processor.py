"""Module for processing files and directories with advanced features."""

import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Set, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class FileType(Enum):
    """Enumeration of supported file types."""
    PDF = "pdf"
    DJVU = "djvu"
    IMAGE = "image"
    DOCUMENT = "document"
    PRESENTATION = "presentation"
    SPREADSHEET = "spreadsheet"
    TEXT = "text"
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
        FileType.TEXT: {'.txt', '.md'},
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
        recursive: bool = True, 
        file_filter: Optional[str] = None
    ) -> Dict[FileType, List[Path]]:
        """Scan directory and identify files by type.
        
        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories recursively
            file_filter: Optional regex pattern to filter filenames
            
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
            
        files_by_type = {file_type: [] for file_type in FileType}
        
        try:
            if recursive:
                files = [f for f in directory.rglob("*") if f.is_file()]
            else:
                files = [f for f in directory.iterdir() if f.is_file()]
                
            # Apply file filter if provided
            if file_filter:
                pattern = re.compile(file_filter)
                files = [f for f in files if pattern.match(f.name)]
                
            # Group files by type
            for file_path in files:
                file_type = self._identify_file_type(file_path)
                files_by_type[file_type].append(file_path)
                
            # Log results
            total_files = sum(len(files) for files in files_by_type.values())
            logger.info(f"Scanned {total_files} files in {directory}")
            
            return files_by_type
            
        except PermissionError:
            raise PermissionError(f"No permission to access directory: {directory}")
        except Exception as e:
            logger.exception(f"Error scanning directory {directory}: {e}")
            raise
    
    def _identify_file_type(self, file_path: Path) -> FileType:
        """Identify file type by extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            FileType: Identified file type
        """
        ext = file_path.suffix.lower()
        
        for file_type, extensions in self.FILE_EXTENSIONS.items():
            if ext in extensions:
                return file_type
                
        return FileType.UNKNOWN
    
    def filter_files_by_pattern(self, files: List[Path], pattern: str) -> List[Path]:
        """Filter files by regex pattern.
        
        Args:
            files: List of file paths
            pattern: Regex pattern to match
            
        Returns:
            List[Path]: Filtered file paths
        """
        try:
            compiled_pattern = re.compile(pattern)
            return [f for f in files if compiled_pattern.match(f.name)]
        except re.error as e:
            logger.error(f"Invalid regex pattern '{pattern}': {e}")
            return files  # Return original list if pattern is invalid
    
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
                    
                    results.append(FileProcessingResult(
                        file_path=file_path,
                        output_path=output_path,
                        success=True,
                        error_message=None,
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
        """Process file with MinerU (placeholder implementation).
        
        Args:
            file_path: Path to file
            file_type: File type
            output_dir: Output directory
            
        Returns:
            Optional[Path]: Output file path or None if failed
        """
        # This is a placeholder - in real implementation, we would call MinerU
        # For now, we'll just create a placeholder output file
        output_file = output_dir / f"{file_path.stem}.md"
        
        try:
            # Create placeholder content
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"# {file_path.name}\n\n")
                f.write(f"This is a placeholder for {file_type.value} file processed from {file_path}\n")
                
            return output_file
        except Exception as e:
            logger.error(f"Error creating placeholder for {file_path}: {e}")
            return None
    
    def get_statistics(self, files_by_type: Dict[FileType, List[Path]]) -> Dict[str, int]:
        """Get statistics about files by type.
        
        Args:
            files_by_type: Files grouped by type
            
        Returns:
            Dict[str, int]: Statistics
        """
        stats = {}
        total = 0
        
        for file_type, files in files_by_type.items():
            count = len(files)
            stats[file_type.value] = count
            total += count
            
        stats['total'] = total
        return stats