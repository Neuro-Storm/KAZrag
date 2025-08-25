"""Test runner for KAZrag project."""

import sys
import os
from pathlib import Path
import subprocess

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_tests():
    """Run all tests for the KAZrag project."""
    print("Running KAZrag tests...")
    
    # Change to project directory
    os.chdir(project_root)
    
    # Run pytest with coverage
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests", 
            "-v", 
            "--tb=short",
            "--cov=core",
            "--cov=web",
            "--cov=config",
            "--cov-report=term-missing"
        ], check=True)
        
        print("\nAll tests completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nTests failed with return code {e.returncode}")
        return False

def run_unit_tests():
    """Run only unit tests."""
    print("Running unit tests...")
    
    # Change to project directory
    os.chdir(project_root)
    
    # Run only unit tests (exclude integration tests)
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/test_config_manager.py",
            "tests/test_collection_manager.py", 
            "tests/test_embedding_manager.py",
            "tests/test_indexer_components.py",
            "tests/test_search_components.py",
            "tests/test_web_routes.py",
            "-v", 
            "--tb=short"
        ], check=True)
        
        print("\nUnit tests completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nUnit tests failed with return code {e.returncode}")
        return False

def run_integration_tests():
    """Run only integration tests."""
    print("Running integration tests...")
    
    # Change to project directory
    os.chdir(project_root)
    
    # Run only integration tests
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/test_indexing_integration.py",
            "tests/test_search_integration.py",
            "tests/test_web_api_integration.py",
            "-v", 
            "--tb=short"
        ], check=True)
        
        print("\nIntegration tests completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nIntegration tests failed with return code {e.returncode}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "unit":
            success = run_unit_tests()
        elif sys.argv[1] == "integration":
            success = run_integration_tests()
        else:
            print("Usage: python test_runner.py [unit|integration]")
            sys.exit(1)
    else:
        success = run_tests()
    
    sys.exit(0 if success else 1)