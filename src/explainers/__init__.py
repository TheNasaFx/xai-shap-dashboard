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

def __getattr__(name):
    if name == "SHAPExplainer":
        from src.explainers.shap_explainer import SHAPExplainer
        return SHAPExplainer
    elif name in ("LocalExplanation", "GlobalExplanation", "InteractionExplanation"):
        from src.explainers import explanation_types
        return getattr(explanation_types, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SHAPExplainer",
    "LocalExplanation",
    "GlobalExplanation",
    "InteractionExplanation",
]
