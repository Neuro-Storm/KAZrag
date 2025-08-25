"""Module for centralized configuration management."""

import json
import time
import logging
from pathlib import Path
from typing import Optional, Any
from pydantic import BaseModel, ValidationError

from config.settings import Config, CONFIG_FILE, DEFAULT_CACHE_TTL

logger = logging.getLogger(__name__)


class ConfigManager:
    """Centralized configuration manager with caching and validation."""
    
    _instance: Optional['ConfigManager'] = None
    
    def __init__(self, cache_ttl: int = DEFAULT_CACHE_TTL):
        """Initialize ConfigManager.
        
        Args:
            cache_ttl: Time to live for cached configuration in seconds
        """
        self.cache_ttl = cache_ttl
        self._cached_config: Optional[Config] = None
        self._cached_config_time: float = 0
        
    @classmethod
    def get_instance(cls, cache_ttl: int = DEFAULT_CACHE_TTL) -> 'ConfigManager':
        """Get singleton instance of ConfigManager.
        
        Args:
            cache_ttl: Time to live for cached configuration in seconds
            
        Returns:
            ConfigManager: Singleton instance
        """
        if cls._instance is None:
            cls._instance = cls(cache_ttl)
        return cls._instance
    
    def load(self) -> Config:
        """Load configuration from file with caching.
        
        Returns:
            Config: Loaded configuration object
        """
        current_time = time.time()
        
        # Check if we have a valid cached config
        if (self._cached_config is not None and 
            current_time - self._cached_config_time <= self.cache_ttl):
            logger.debug("Returning cached configuration")
            return self._cached_config
            
        # Load config from file
        logger.debug("Loading configuration from file")
        if not CONFIG_FILE.exists():
            # Create default config if file doesn't exist
            config = Config()
            self.save(config)
            self._cached_config = config
            self._cached_config_time = current_time
            return self._cached_config
            
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
                
            # Create Config object with validation
            config = Config(**config_dict)
            
            # Update cache
            self._cached_config = config
            self._cached_config_time = current_time
            
            return config
            
        except ValidationError as e:
            logger.exception(f"Configuration validation errors: {e.errors()}")
            # Return default config if validation fails
            config = Config()
            self.save(config)
            self._cached_config = config
            self._cached_config_time = current_time
            return self._cached_config
            
        except Exception as e:
            logger.exception(f"Error loading configuration: {e}")
            # Return default config if any other error occurs
            config = Config()
            self.save(config)
            self._cached_config = config
            self._cached_config_time = current_time
            return self._cached_config
    
    def save(self, config: Config) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration object to save
        """
        try:
            # Convert Config to dict
            config_dict = config.model_dump()
            
            # Save to file
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=4, ensure_ascii=False)
                
            # Update cache
            self._cached_config = config
            self._cached_config_time = time.time()
            
            logger.info("Configuration saved successfully")
            
        except Exception as e:
            logger.exception(f"Error saving configuration: {e}")
            raise
    
    def get(self) -> Config:
        """Get configuration (alias for load).
        
        Returns:
            Config: Configuration object
        """
        return self.load()
    
    def reload(self) -> Config:
        """Force reload configuration from file, bypassing cache.
        
        Returns:
            Config: Reloaded configuration object
        """
        # Clear cache
        self._cached_config = None
        self._cached_config_time = 0
        
        # Load fresh config
        return self.load()
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a specific configuration value by key.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Any: Configuration value or default
        """
        config = self.get()
        return getattr(config, key, default)
    
    def set_value(self, key: str, value: Any) -> None:
        """Set a specific configuration value by key.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        config = self.get()
        if hasattr(config, key):
            setattr(config, key, value)
            self.save(config)
        else:
            raise AttributeError(f"Configuration has no attribute '{key}'")