#!/bin/bash
# CI/CD test script for KAZrag

echo "Starting KAZrag CI/CD tests..."

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run unit tests
echo "Running unit tests..."
python -m pytest tests/test_config_manager.py tests/test_collection_manager.py tests/test_embedding_manager.py tests/test_indexer_components.py tests/test_search_components.py tests/test_web_routes.py -v --tb=short

# Check if unit tests passed
if [ $? -ne 0 ]; then
    echo "Unit tests failed!"
    exit 1
fi

# Run integration tests
echo "Running integration tests..."
python -m pytest tests/test_indexing_integration.py tests/test_search_integration.py tests/test_web_api_integration.py -v --tb=short

# Check if integration tests passed
if [ $? -ne 0 ]; then
    echo "Integration tests failed!"
    exit 1
fi

# Run all tests with coverage
echo "Running all tests with coverage..."
python -m pytest tests --cov=core --cov=web --cov=config --cov-report=xml --cov-report=term-missing

# Check if all tests passed
if [ $? -ne 0 ]; then
    echo "Some tests failed!"
    exit 1
fi

echo "All tests passed successfully!"
exit 0