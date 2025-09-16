"""Module for resolving resource paths in both development and packaged environments."""

import sys
from pathlib import Path


def resource_path(relative_path: str) -> Path:
    """Get absolute path to resource, works for dev and for PyInstaller bundles.
    
    Args:
        relative_path: Relative path to the resource
        
    Returns:
        Absolute path to the resource
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = Path(sys._MEIPASS)
    except Exception:
        # In development environment, use the current directory
        base_path = Path(__file__).resolve().parent.parent

    return base_path / relative_path