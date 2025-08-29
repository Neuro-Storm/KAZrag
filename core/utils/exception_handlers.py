"""Module for centralized exception handling in the FastAPI application."""

import logging
import uuid
import traceback
import os
from typing import Union
from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from jinja2 import TemplateNotFound

logger = logging.getLogger(__name__)


class AppException(Exception):
    """Base application exception."""
    def __init__(self, message: str, status_code: int = 500, error_code: str = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(self.message)


class ValidationError(AppException):
    """Validation error exception."""
    def __init__(self, message: str):
        super().__init__(message, status_code=400, error_code="VALIDATION_ERROR")


class AuthenticationError(AppException):
    """Authentication error exception."""
    def __init__(self, message: str):
        super().__init__(message, status_code=401, error_code="AUTHENTICATION_ERROR")


class AuthorizationError(AppException):
    """Authorization error exception."""
    def __init__(self, message: str):
        super().__init__(message, status_code=403, error_code="AUTHORIZATION_ERROR")


class NotFoundError(AppException):
    """Not found error exception."""
    def __init__(self, message: str):
        super().__init__(message, status_code=404, error_code="NOT_FOUND_ERROR")


def add_exception_handlers(app: FastAPI) -> None:
    """Add centralized exception handlers to the FastAPI application.
    
    Args:
        app (FastAPI): The FastAPI application instance
    """
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        """Handle custom application exceptions."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        # Log the error with request context
        logger.error(
            f"AppException [{request_id}]: {exc.message} "
            f"(status_code={exc.status_code}, error_code={exc.error_code}) "
            f"Path: {request.url.path}, Method: {request.method}"
        )
        
        # For AJAX requests, return JSON response
        if request.headers.get("accept", "").startswith("application/json") or \
           request.headers.get("content-type", "").startswith("application/json"):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "type": exc.error_code or "APPLICATION_ERROR",
                        "message": exc.message,
                        "request_id": request_id
                    }
                }
            )
        
        # For HTML requests, return error page or redirect with error message
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "type": exc.error_code or "APPLICATION_ERROR",
                    "message": exc.message,
                    "request_id": request_id
                }
            }
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        # Log the error with request context
        logger.warning(
            f"HTTPException [{request_id}]: {exc.detail} "
            f"(status_code={exc.status_code}) "
            f"Path: {request.url.path}, Method: {request.method}"
        )
        
        # Special handling for 401 Unauthorized in admin routes
        if exc.status_code == 401 and request.url.path.startswith("/api/admin/"):
            # For AJAX requests, return JSON response
            if request.headers.get("accept", "").startswith("application/json") or \
               request.headers.get("content-type", "").startswith("application/json"):
                return JSONResponse(
                    status_code=exc.status_code,
                    content={
                        "error": {
                            "type": "AUTHENTICATION_ERROR",
                            "message": "Требуется аутентификация. Используйте HTTP Basic Authentication с API ключом из .env файла в качестве пароля.",
                            "request_id": request_id
                        }
                    }
                )
            
            # For HTML requests to admin routes, return a more informative error page
            from fastapi.responses import HTMLResponse
            return HTMLResponse(
                content=f"""
                <html>
                    <head><title>Требуется аутентификация</title></head>
                    <body>
                        <h1>Требуется аутентификация</h1>
                        <p>Для доступа к этой странице необходимо войти в систему.</p>
                        <p>Используйте HTTP Basic Authentication:</p>
                        <ul>
                            <li>Имя пользователя: любое</li>
                            <li>Пароль: значение переменной ADMIN_API_KEY из файла .env</li>
                        </ul>
                        <p>Текущее значение ADMIN_API_KEY: {'установлено' if os.getenv('ADMIN_API_KEY') else 'не установлено'}</p>
                        <p><a href="/">Вернуться на главную</a></p>
                    </body>
                </html>
                """,
                status_code=401,
                headers={"WWW-Authenticate": "Basic"}
            )
        
        # For AJAX requests, return JSON response
        if request.headers.get("accept", "").startswith("application/json") or \
           request.headers.get("content-type", "").startswith("application/json"):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "type": "HTTP_ERROR",
                        "message": exc.detail,
                        "request_id": request_id
                    }
                }
            )
        
        # For HTML requests, we let the default handler take over for now
        # In a real implementation, we might render an error template
        # Instead of raising the exception, we return a proper 404 response
        from fastapi.responses import HTMLResponse
        return HTMLResponse(
            content=f"""
            <html>
                <head><title>404 Not Found</title></head>
                <body>
                    <h1>404 Not Found</h1>
                    <p>The requested URL was not found on this server.</p>
                    <p>Path: {request.url.path}</p>
                </body>
            </html>
            """,
            status_code=404
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        # Log the error with request context
        logger.warning(
            f"ValidationError [{request_id}]: {exc.errors()} "
            f"Path: {request.url.path}, Method: {request.method}"
        )
        
        # Format validation errors for better readability
        errors = []
        for error in exc.errors():
            errors.append({
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })
        
        # For AJAX requests, return JSON response
        if request.headers.get("accept", "").startswith("application/json") or \
           request.headers.get("content-type", "").startswith("application/json"):
            return JSONResponse(
                status_code=422,
                content={
                    "error": {
                        "type": "VALIDATION_ERROR",
                        "message": "Validation failed",
                        "details": errors,
                        "request_id": request_id
                    }
                }
            )
        
        # For HTML requests, return JSON for now (would render template in full implementation)
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "type": "VALIDATION_ERROR",
                    "message": "Validation failed",
                    "details": errors,
                    "request_id": request_id
                }
            }
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Handle all other unhandled exceptions."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        # Log the error with full traceback
        logger.exception(
            f"UnhandledException [{request_id}]: {str(exc)} "
            f"Path: {request.url.path}, Method: {request.method}"
        )
        
        # For AJAX requests, return JSON response
        if request.headers.get("accept", "").startswith("application/json") or \
           request.headers.get("content-type", "").startswith("application/json"):
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "type": "INTERNAL_SERVER_ERROR",
                        "message": "An unexpected error occurred",
                        "request_id": request_id
                    }
                }
            )
        
        # For HTML requests, return JSON for now (would render template in full implementation)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred",
                    "request_id": request_id
                }
            }
        )


def get_request_id(request: Request) -> str:
    """Get or generate a request ID for logging context.
    
    Args:
        request (Request): The FastAPI request object
        
    Returns:
        str: Request ID
    """
    if not hasattr(request.state, "request_id"):
        request.state.request_id = str(uuid.uuid4())
    return request.state.request_id