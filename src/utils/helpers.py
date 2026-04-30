"""
Helper Functions - Common Utilities
====================================

Author: XAI-SHAP Framework
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np
import yaml


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the framework.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
        format_string: Custom format string
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers
    )
    
    return logging.getLogger("xai_shap")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        path: Output file path
        indent: JSON indentation
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: Union[str, Path]) -> Any:
    """
    Load data from JSON file.
    
    Args:
        path: Input file path
        
    Returns:
        Loaded data
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    
    # Navigate up to find project root (contains src/)
    for parent in current.parents:
        if (parent / 'src').exists() or (parent / 'README.md').exists():
            return parent
    
    return current.parent.parent.parent


def format_number(value: float, precision: int = 4) -> str:
    """Format number for display."""
    if abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.{precision}f}"


def truncate_string(s: str, max_length: int = 50) -> str:
    """Truncate string with ellipsis."""
    if len(s) <= max_length:
        return s
    return s[:max_length - 3] + "..."


def unwrap_model_for_shap(model: Any) -> Tuple[Any, str]:
    """Return the concrete estimator that SHAP should inspect."""
    actual_model = model

    if hasattr(actual_model, 'model'):
        try:
            nested_model = actual_model.model
        except Exception:
            nested_model = None

        if nested_model is not None:
            actual_model = nested_model

    model_type = type(actual_model).__name__.lower()
    if 'pipeline' in model_type and hasattr(actual_model, 'steps') and actual_model.steps:
        actual_model = actual_model.steps[-1][1]
        model_type = type(actual_model).__name__.lower()

    return actual_model, model_type


def get_shap_prediction_callable(
    model: Any,
    actual_model: Optional[Any] = None
) -> Callable[..., Any]:
    """Return a callable prediction function for SHAP fallback explainers."""
    actual_model = actual_model if actual_model is not None else unwrap_model_for_shap(model)[0]
    prediction_model = model if hasattr(model, 'predict') else actual_model

    if hasattr(prediction_model, 'predict_proba'):
        def predict_fn(X):
            probabilities = prediction_model.predict_proba(X)
            if probabilities is None:
                return prediction_model.predict(X)
            return probabilities

        return predict_fn

    if hasattr(prediction_model, 'predict'):
        return prediction_model.predict

    if callable(actual_model):
        return actual_model

    raise TypeError(f"SHAP callable үүсгэж чадсангүй: {type(model).__name__}")


def get_binary_probability_scores(model: Any, X: np.ndarray) -> Optional[np.ndarray]:
    """Return positive-class probabilities for binary classifiers when available."""
    actual_model = unwrap_model_for_shap(model)[0]
    prediction_model = model if hasattr(model, 'predict_proba') else actual_model

    if not hasattr(prediction_model, 'predict_proba'):
        return None

    probabilities = np.asarray(prediction_model.predict_proba(X))
    if probabilities.ndim == 1:
        return probabilities.astype(float)
    if probabilities.ndim == 2 and probabilities.shape[1] == 1:
        return probabilities[:, 0].astype(float)
    if probabilities.ndim == 2 and probabilities.shape[1] == 2:
        return probabilities[:, 1].astype(float)

    return None


def predict_with_threshold(
    model: Any,
    X: np.ndarray,
    decision_threshold: Optional[float] = None,
) -> np.ndarray:
    """Predict labels, optionally overriding the default 0.5 decision threshold."""
    if decision_threshold is None:
        return np.asarray(model.predict(X))

    probability_scores = get_binary_probability_scores(model, X)
    if probability_scores is None:
        return np.asarray(model.predict(X))

    return (probability_scores >= float(decision_threshold)).astype(int)


def create_shap_explainer(model: Any, background: np.ndarray) -> Any:
    """Create a SHAP explainer that works with wrapper models."""
    import shap

    actual_model, model_type = unwrap_model_for_shap(model)
    prediction_callable = get_shap_prediction_callable(model, actual_model)
    tree_types = [
        'xgb', 'lgb', 'lightgbm', 'catboost',
        'randomforest', 'extratrees', 'decisiontree',
        'gradientboosting', 'adaboost', 'bagging'
    ]

    if any(tree_type in model_type for tree_type in tree_types):
        try:
            return shap.TreeExplainer(actual_model)
        except Exception:
            pass

    return shap.Explainer(prediction_callable, background)


def extract_shap_values(explainer: Any, X: np.ndarray) -> np.ndarray:
    """Normalize SHAP values across old and new SHAP APIs."""
    if hasattr(explainer, 'shap_values'):
        shap_values = explainer.shap_values(X)
    else:
        shap_values = explainer(X).values

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    return np.asarray(shap_values)
