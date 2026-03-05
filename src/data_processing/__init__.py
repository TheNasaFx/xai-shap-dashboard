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

from src.data_processing.processor import DataProcessor
from src.data_processing.bias_detector import BiasDetector
from src.data_processing.transformers import (
    MissingValueHandler,
    CategoricalEncoder,
    FeatureNormalizer
)

__all__ = [
    "DataProcessor",
    "BiasDetector",
    "MissingValueHandler",
    "CategoricalEncoder",
    "FeatureNormalizer",
]
