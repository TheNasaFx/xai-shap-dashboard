"""
Core module for XAI-SHAP Framework
"""


def __getattr__(name):
    if name == "XAIFramework":
        from src.core.framework import XAIFramework
        return XAIFramework
    elif name == "XAIPipeline":
        from src.core.pipeline import XAIPipeline
        return XAIPipeline
    elif name == "ConfigManager":
        from src.core.config_manager import ConfigManager
        return ConfigManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "XAIFramework",
    "XAIPipeline", 
    "ConfigManager",
]
