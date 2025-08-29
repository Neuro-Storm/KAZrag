"""Module for converting HTML files to Markdown format."""

import logging
from pathlib import Path
from typing import List
try:
    import html2text
    HAS_HTML2TEXT = True
except ImportError:
    HAS_HTML2TEXT = False

from .base import BaseConverter

logger = logging.getLogger(__name__)


class HtmlConverter(BaseConverter):
    """Converter for HTML files to Markdown format."""
    
    def convert(self, file_path: Path, output_dir: Path) -> List[Path]:
        """
        Convert an HTML file to Markdown format.
        
        Args:
            file_path (Path): Path to the HTML file
            output_dir (Path): Directory to save the output files
            
        Returns:
            List[Path]: List of paths to the converted Markdown files
        """
        if not HAS_HTML2TEXT:
            raise ImportError("html2text is required for HTML conversion. Please install it with 'pip install html2text'")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read HTML content
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Convert HTML to Markdown
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.body_width = 0  # Don't wrap lines
        md_content = h.handle(html_content)
        
        # Save to file
        output_file = output_dir / f"{file_path.stem}.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        logger.info(f"Converted {file_path.name} to {output_file}")
        return [output_file]
    
    def supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List[str]: List of supported extensions
        """
        return ['.html', '.htm']