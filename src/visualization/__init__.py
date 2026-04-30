"""
Visualization Module
=====================

Interactive visualization components for SHAP explanations.
Supports multiple visualization libraries and plot types.

Author: XAI-SHAP Framework
"""

def __getattr__(name):
    if name == "XAIVisualizer":
        from src.visualization.plots import XAIVisualizer
        return XAIVisualizer
    elif name == "InteractivePlots":
        from src.visualization.interactive import InteractivePlots
        return InteractivePlots
    elif name == "MetricsVisualizer":
        from src.visualization.metrics_viz import MetricsVisualizer
        return MetricsVisualizer
    elif name in ("PlotTheme", "ThemeManager"):
        from src.visualization import themes
        return getattr(themes, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "XAIVisualizer",
    "InteractivePlots",
    "MetricsVisualizer",
    "PlotTheme",
    "ThemeManager",
]
