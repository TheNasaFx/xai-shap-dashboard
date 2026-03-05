"""
Evaluation Module
==================

Model evaluation and Responsible AI metrics.

Author: XAI-SHAP Framework
"""

def __getattr__(name):
    if name == "ModelEvaluator":
        from src.evaluation.metrics import ModelEvaluator
        return ModelEvaluator
    elif name == "FairnessEvaluator":
        from src.evaluation.fairness import FairnessEvaluator
        return FairnessEvaluator
    elif name == "BiasDetector":
        from src.evaluation.fairness import BiasDetector
        return BiasDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ModelEvaluator",
    "FairnessEvaluator",
    "BiasDetector",
]
