@echo off
REM CI/CD test script for KAZrag (Windows version)

echo Starting KAZrag CI/CD tests...

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Run unit tests
echo Running unit tests...
python -m pytest tests/test_config_manager.py tests/test_collection_manager.py tests/test_embedding_manager.py tests/test_indexer_components.py tests/test_search_components.py tests/test_web_routes.py -v --tb=short

REM Check if unit tests passed
if %ERRORLEVEL% NEQ 0 (
    echo Unit tests failed!
    exit /b 1
)

REM Run integration tests
echo Running integration tests...
python -m pytest tests/test_indexing_integration.py tests/test_search_integration.py tests/test_web_api_integration.py -v --tb=short

REM Check if integration tests passed
if %ERRORLEVEL% NEQ 0 (
    echo Integration tests failed!
    exit /b 1
)

REM Run all tests with coverage
echo Running all tests with coverage...
python -m pytest tests --cov=core --cov=web --cov=config --cov-report=xml --cov-report=term-missing

REM Check if all tests passed
if %ERRORLEVEL% NEQ 0 (
    echo Some tests failed!
    exit /b 1
)

echo All tests passed successfully!
exit /b 0