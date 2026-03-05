"""
Dashboard Module
=================

Streamlit-based interactive dashboard for visual analytics.

Author: XAI-SHAP Framework
"""

from src.dashboard.app import run_dashboard, create_dashboard
from src.dashboard.components import (
    SidebarComponent,
    MetricsComponent,
    ExplanationComponent
)

__all__ = [
    "run_dashboard",
    "create_dashboard",
    "SidebarComponent",
    "MetricsComponent",
    "ExplanationComponent",
]
