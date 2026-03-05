"""
Evaluation Module
==================

Model evaluation and Responsible AI metrics.

Author: XAI-SHAP Framework
"""

from src.evaluation.metrics import ModelEvaluator
from src.evaluation.fairness import FairnessEvaluator, BiasDetector

__all__ = [
    "ModelEvaluator",
    "FairnessEvaluator",
    "BiasDetector",
]
