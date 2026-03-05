"""
Explainers Module - SHAP-based Explanation Generation
======================================================

This module provides the core SHAP explanation functionality:
- Multiple SHAP explainer types (Tree, Kernel, Deep, Linear)
- Local explanations (individual predictions)
- Global explanations (overall model behavior)
- Feature interaction analysis

Author: XAI-SHAP Framework
"""

from src.explainers.shap_explainer import SHAPExplainer
from src.explainers.explanation_types import (
    LocalExplanation,
    GlobalExplanation,
    InteractionExplanation
)

__all__ = [
    "SHAPExplainer",
    "LocalExplanation",
    "GlobalExplanation",
    "InteractionExplanation",
]
