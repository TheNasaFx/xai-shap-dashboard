"""
XAI-SHAP Visual Analytics Framework
====================================

A comprehensive framework for explaining black-box machine learning models
using SHAP (SHapley Additive exPlanations) with interactive visual analytics.

This framework supports:
- Multiple ML models (XGBoost, Random Forest, Neural Networks)
- SHAP-based local and global explanations
- Interactive visualization dashboard
- Responsible AI evaluation (fairness, transparency)

Author: Bachelor's Thesis - Computer Science 2025/2026
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Diploma Project"
__description__ = "Visual Analytics Framework for Explainable AI using SHAP"

from src.core.framework import XAIFramework
from src.core.pipeline import XAIPipeline

__all__ = [
    "XAIFramework",
    "XAIPipeline",
    "__version__",
]
