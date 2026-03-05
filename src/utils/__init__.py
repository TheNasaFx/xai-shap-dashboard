"""
Utility Functions Module
=========================

Common utility functions for the framework.

Author: XAI-SHAP Framework
"""

def __getattr__(name):
    if name in ("setup_logging", "load_config", "save_json", "load_json"):
        from src.utils import helpers
        return getattr(helpers, name)
    elif name == "ReportGenerator":
        from src.utils.reporting import ReportGenerator
        return ReportGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "setup_logging",
    "load_config",
    "save_json",
    "load_json",
    "ReportGenerator",
]
