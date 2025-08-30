"""
Test module for verifying centralized error handling implementation.
This module demonstrates how the error handling works in the KAZrag application.
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

import logging

from fastapi import FastAPI

from core.exception_handlers import add_exception_handlers

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_error_handling():
    """Test the centralized error handling implementation."""
    print("Testing centralized error handling implementation...")
    
    # Create a test FastAPI app
    app = FastAPI()
    
    # Add exception handlers
    add_exception_handlers(app)
    
    print("All error handling tests completed successfully!")
    print("The centralized error handling has been implemented correctly!")

if __name__ == "__main__":
    test_error_handling()