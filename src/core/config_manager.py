"""
Configuration Manager
======================

Handles loading and managing framework configuration from YAML files.
Provides centralized access to all configuration settings.

Author: XAI-SHAP Framework
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Centralized configuration management for the XAI-SHAP Framework.
    
    This class handles:
    - Loading configuration from YAML files
    - Providing default values for missing settings
    - Validating configuration parameters
    - Runtime configuration updates
    
    Example:
        >>> config = ConfigManager()
        >>> model_config = config.get("models.xgboost")
        >>> config.set("shap.max_display_features", 15)
    """
    
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        """Singleton pattern to ensure single config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize ConfigManager with default configuration."""
        if self._initialized:
            return
            
        self._initialized = True
        self._config_path = self._find_config_file()
        self._load_config()
        self._apply_defaults()
        logger.info("ConfigManager initialized successfully")
    
    def _find_config_file(self) -> Path:
        """Find the configuration file in standard locations."""
        possible_paths = [
            Path("config/config.yaml"),
            Path("../config/config.yaml"),
            Path(__file__).parent.parent.parent / "config" / "config.yaml",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # Return default path even if doesn't exist
        return possible_paths[0]
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            if self._config_path.exists():
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f) or {}
                logger.info(f"Configuration loaded from {self._config_path}")
            else:
                logger.warning(f"Config file not found at {self._config_path}, using defaults")
                self._config = {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self._config = {}
    
    def _apply_defaults(self) -> None:
        """Apply default values for missing configuration."""
        defaults = {
            "app": {
                "name": "XAI-SHAP Visual Analytics",
                "version": "1.0.0",
                "debug": False,
                "log_level": "INFO"
            },
            "data_processing": {
                "normalization_method": "standard",
                "missing_value_strategy": "median",
                "categorical_encoding": "onehot",
                "test_size": 0.2,
                "random_state": 42
            },
            "models": {
                "default_model": "xgboost"
            },
            "shap": {
                "background_samples": 100,
                "max_display_features": 20,
                "explanation_type": "both",
                "explainer_type": "auto"
            },
            "visualization": {
                "default_theme": "plotly",
                "interactive": True,
                "figure_width": 10,
                "figure_height": 6
            },
            "responsible_ai": {
                "bias_detection": True,
                "protected_attributes": []
            }
        }
        
        self._config = self._deep_merge(defaults, self._config)
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "models.xgboost.n_estimators")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            >>> config.get("shap.max_display_features", 20)
            20
        """
        keys = key.split(".")
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "shap.max_display_features")
            value: Value to set
        """
        keys = key.split(".")
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.debug(f"Configuration updated: {key} = {value}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., "models", "shap")
            
        Returns:
            Dictionary containing section configuration
        """
        return self._config.get(section, {})
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()
        self._apply_defaults()
        logger.info("Configuration reloaded")
    
    def save(self, path: Optional[Path] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            path: Path to save configuration (default: original path)
        """
        save_path = path or self._config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Configuration saved to {save_path}")
    
    @property
    def all(self) -> Dict[str, Any]:
        """Return complete configuration dictionary."""
        return self._config.copy()
    
    def __repr__(self) -> str:
        return f"ConfigManager(path={self._config_path})"


# Global config instance
config = ConfigManager()
