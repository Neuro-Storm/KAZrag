"""Module for centralized configuration management using pydantic-settings."""

import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from cachetools import TTLCache
from pydantic import ValidationError

from config.settings_model import Config
from config.resource_path import resource_path

logger = logging.getLogger(__name__)


class ConfigManager:
    """Centralized configuration manager with caching and validation using pydantic-settings."""
    
    _instance: Optional['ConfigManager'] = None
    
    def __init__(self, cache_ttl: int = 60):
        """Initialize ConfigManager.
        
        Args:
            cache_ttl: Time to live for cached configuration in seconds
        """
        self.cache_ttl = cache_ttl
        # Use TTLCache for configuration caching
        self._cache = TTLCache(maxsize=1, ttl=cache_ttl)
        self._config_instance: Optional[Config] = None
        
    @classmethod
    def get_instance(cls, cache_ttl: int = 60) -> 'ConfigManager':
        """Get singleton instance of ConfigManager.
        
        Args:
            cache_ttl: Time to live for cached configuration in seconds
            
        Returns:
            ConfigManager: Singleton instance
        """
        if cls._instance is None:
            cls._instance = cls(cache_ttl)
        return cls._instance
    
    def _load_config_from_file(self) -> Config:
        """Load configuration from file without caching.
        
        Returns:
            Config: Loaded configuration object
        """
        logger.debug("Loading configuration from file")
        config_file = resource_path("config/config.json")
        
        if not config_file.exists():
            # Create default config if file doesn't exist
            config = Config()
            self.save(config)
            return config
            
        try:
            with open(config_file, encoding="utf-8") as f:
                config_dict = json.load(f)
                
            # Create Config object with validation
            config = Config(**config_dict)
            return config
            
        except ValidationError as e:
            logger.exception(f"Configuration validation errors: {e.errors()}")
            # Return default config if validation fails
            config = Config()
            self.save(config)
            return config
            
        except Exception as e:
            logger.exception(f"Error loading configuration: {e}")
            # Return default config if any other error occurs
            config = Config()
            self.save(config)
            return config
    
    def _load_config_from_settings(self) -> Config:
        """Load configuration from environment variables and settings.
        
        Returns:
            Config: Loaded configuration object
        """
        try:
            # Load config from environment variables and defaults
            config = Config()
            return config
        except Exception as e:
            logger.exception(f"Error loading configuration from settings: {e}")
            # Fallback to file-based config
            return self._load_config_from_file()
    
    def load(self) -> Config:
        """Load configuration from file with caching.
        
        Returns:
            Config: Loaded configuration object
        """
        # Check if we have a valid cached config
        if 'config' in self._cache:
            logger.debug("Returning cached configuration")
            return self._cache['config']
        
        # Try to load from file first
        try:
            config = self._load_config_from_file()
        except Exception as e:
            logger.warning(f"Failed to load config from file, falling back to settings: {e}")
            # Fallback to settings-based config
            config = self._load_config_from_settings()
        
        # Update cache
        self._cache['config'] = config
        
        return config
    
    def save(self, config: Config) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration object to save
        """
        try:
            # Save to file
            config.save_to_file()
            
            # Update cache
            self._cache['config'] = config
            
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
        # Clear the cache
        self._cache.clear()
        
        # Load fresh config
        return self.load()
    
    def update_from_dict(self, updates: Dict[str, Any]) -> Config:
        """Update configuration from dictionary.
        
        Args:
            updates: Dictionary with configuration updates
            
        Returns:
            Config: Updated configuration object
        """
        try:
            # Get current config
            current_config = self.get()
            
            # Convert to dict and update
            config_dict = current_config.model_dump()
            config_dict.update(updates)
            
            # Create new config object
            updated_config = Config(**config_dict)
            
            # Save updated config
            self.save(updated_config)
            
            return updated_config
            
        except Exception as e:
            logger.exception(f"Error updating configuration: {e}")
            raise