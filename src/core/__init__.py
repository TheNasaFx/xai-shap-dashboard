"""
Core module for XAI-SHAP Framework
"""

from src.core.framework import XAIFramework
from src.core.pipeline import XAIPipeline
from src.core.config_manager import ConfigManager

__all__ = [
    "XAIFramework",
    "XAIPipeline", 
    "ConfigManager",
]
