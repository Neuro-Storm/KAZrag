"""Unit tests for the ConfigManager class."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open
from config.config_manager import ConfigManager
from config.settings import Config


class TestConfigManager:
    """Test suite for ConfigManager class."""

    def test_singleton_instance(self):
        """Test that ConfigManager is a singleton."""
        instance1 = ConfigManager.get_instance()
        instance2 = ConfigManager.get_instance()
        assert instance1 is instance2

    @patch('config.config_manager.CONFIG_FILE')
    def test_load_default_config(self, mock_config_file):
        """Test loading default configuration."""
        # Mock the Path object methods
        mock_config_file.exists.return_value = False
        
        manager = ConfigManager()
        config = manager.load()
        assert isinstance(config, Config)
        assert config.folder_path == "./data_to_index"
        assert config.collection_name == "final-dense-collection"

    @patch('config.config_manager.CONFIG_FILE')
    def test_load_existing_config(self, mock_config_file):
        """Test loading existing configuration from file."""
        # Create a sample config dict
        sample_config = {
            "folder_path": "./test_folder",
            "collection_name": "test-collection",
            "current_hf_model": "test-model"
        }
        
        # Mock the Path object methods
        mock_config_file.exists.return_value = True
        
        with patch('builtins.open', mock_open(read_data=json.dumps(sample_config))):
            manager = ConfigManager()
            config = manager.load()
            assert isinstance(config, Config)
            assert config.folder_path == "./test_folder"
            assert config.collection_name == "test-collection"
            assert config.current_hf_model == "test-model"

    def test_save_config(self):
        """Test saving configuration to file."""
        config = Config()
        config.folder_path = "./new_folder"
        
        with patch('builtins.open', mock_open()) as mock_file:
            manager = ConfigManager()
            manager.save(config)
            mock_file.assert_called_once()
            
    def test_get_value(self):
        """Test getting specific configuration value."""
        manager = ConfigManager()
        with patch.object(manager, 'get') as mock_get:
            mock_config = Config()
            mock_config.folder_path = "./test_path"
            mock_get.return_value = mock_config
            
            value = manager.get_value('folder_path')
            assert value == "./test_path"
            
            # Test default value
            value = manager.get_value('nonexistent_key', 'default')
            assert value == 'default'

    def test_set_value(self):
        """Test setting specific configuration value."""
        manager = ConfigManager()
        mock_config = Config()
        
        with patch.object(manager, 'get') as mock_get:
            with patch.object(manager, 'save') as mock_save:
                mock_get.return_value = mock_config
                manager.set_value('folder_path', './new_path')
                assert mock_config.folder_path == './new_path'
                mock_save.assert_called_once()

    def test_reload_config(self):
        """Test reloading configuration from file."""
        manager = ConfigManager()
        with patch.object(manager, 'load') as mock_load:
            manager.reload()
            mock_load.assert_called_once()