"""
Data Processing Module
=======================

Handles all data preprocessing operations including:
- Loading data from various sources
- Missing value imputation
- Categorical encoding
- Feature normalization
- Bias detection
- Train/test splitting

Author: XAI-SHAP Framework
"""

def __getattr__(name):
    if name == "DataProcessor":
        from src.data_processing.processor import DataProcessor
        return DataProcessor
    elif name == "BiasDetector":
        from src.data_processing.bias_detector import BiasDetector
        return BiasDetector
    elif name in ("MissingValueHandler", "CategoricalEncoder", "FeatureNormalizer"):
        from src.data_processing import transformers
        return getattr(transformers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DataProcessor",
    "BiasDetector",
    "MissingValueHandler",
    "CategoricalEncoder",
    "FeatureNormalizer",
]
