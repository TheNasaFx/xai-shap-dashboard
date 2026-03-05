"""
Base Model - Abstract Base Class for Models
============================================

Provides common interface and utilities for all model implementations.

Author: XAI-SHAP Framework
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for machine learning models.
    
    All model implementations should inherit from this class
    to ensure consistent interface.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self._model = None
        self._is_fitted = False
        self._feature_names = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseModel':
        """
        Fit the model to training data.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional fitting parameters
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions array
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict class probabilities (for classifiers).
        
        Args:
            X: Features to predict
            
        Returns:
            Probability array or None if not supported
        """
        return None
    
    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted
    
    @property
    def feature_names(self) -> Optional[List[str]]:
        """Get feature names if available."""
        return self._feature_names
    
    @feature_names.setter
    def feature_names(self, names: List[str]):
        """Set feature names."""
        self._feature_names = names
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.config.copy()
    
    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters."""
        self.config.update(params)
        return self
    
    @abstractmethod
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get built-in feature importance (if available).
        
        Returns:
            Feature importance array or None
        """
        pass
    
    def __repr__(self) -> str:
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}({fitted_str})"
