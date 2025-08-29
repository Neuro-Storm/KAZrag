"""Module for converting Excel files to Markdown format."""

import logging
from pathlib import Path
from typing import List
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from .base import BaseConverter

logger = logging.getLogger(__name__)


class ExcelConverter(BaseConverter):
    """Converter for Excel files to Markdown format."""
    
    def convert(self, file_path: Path, output_dir: Path) -> List[Path]:
        """
        Convert an Excel file to Markdown format.
        
        Args:
            file_path (Path): Path to the Excel file
            output_dir (Path): Directory to save the output files
            
        Returns:
            List[Path]: List of paths to the converted Markdown files
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for Excel conversion. Please install it with 'pip install pandas'")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read Excel file
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            md_content = self._excel_to_markdown(file_path, excel_file)
        except Exception as e:
            raise RuntimeError(f"Error reading Excel file: {e}")
        
        # Save to file
        output_file = output_dir / f"{file_path.stem}.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        logger.info(f"Converted {file_path.name} to {output_file}")
        return [output_file]
    
    def _excel_to_markdown(self, file_path: Path, excel_file) -> str:
        """Convert Excel file to Markdown format."""
        md_lines = []
        
        # Add title
        md_lines.append(f"# Excel File: {file_path.name}")
        md_lines.append("")
        
        # Process each sheet
        for sheet_name in excel_file.sheet_names:
            md_lines.append(f"## Sheet: {sheet_name}")
            md_lines.append("")
            
            # Read sheet data
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            # Convert to markdown table
            if not df.empty:
                # Handle NaN values
                df = df.fillna("")
                
                # Create markdown table
                md_table = df.to_markdown(index=False)
                md_lines.append(md_table)
            else:
                md_lines.append("*Empty sheet*")
                
            md_lines.append("")
            md_lines.append("---")  # Separator between sheets
            md_lines.append("")
            
        return "\n".join(md_lines)
    
    def supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List[str]: List of supported extensions
        """
        return ['.xlsx', '.xls']