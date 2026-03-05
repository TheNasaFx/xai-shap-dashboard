"""
Dashboard Module
=================

Streamlit-based interactive dashboard for visual analytics.

Author: XAI-SHAP Framework
"""


def __getattr__(name):
    if name in ("run_dashboard", "create_dashboard"):
        from src.dashboard import app
        return getattr(app, name)
    elif name in ("SidebarComponent", "MetricsComponent", "ExplanationComponent"):
        from src.dashboard import components
        return getattr(components, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "run_dashboard",
    "create_dashboard",
    "SidebarComponent",
    "MetricsComponent",
    "ExplanationComponent",
]
