"""
Models Module
==============

Machine learning model implementations including:
- XGBoost (Gradient Boosting)
- LightGBM (Light Gradient Boosting Machine)
- CatBoost (Categorical Boosting)
- Random Forest
- Extra Trees
- Gradient Boosting (sklearn)
- AdaBoost
- Neural Networks (MLP)
- Logistic Regression
- Support Vector Machine (SVM)

Author: XAI-SHAP Framework
"""

from src.models.trainer import ModelTrainer
from src.models.base_model import BaseModel
from src.models.random_forest_model import RandomForestModel

# Conditionally import optional models
try:
    from src.models.xgboost_model import XGBoostModel
except ImportError:
    XGBoostModel = None

try:
    from src.models.neural_network_model import NeuralNetworkModel
except ImportError:
    NeuralNetworkModel = None

__all__ = [
    "ModelTrainer",
    "BaseModel",
    "RandomForestModel",
]

if XGBoostModel is not None:
    __all__.append("XGBoostModel")
if NeuralNetworkModel is not None:
    __all__.append("NeuralNetworkModel")

# Available model types
AVAILABLE_MODELS = [
    "xgboost",
    "lightgbm", 
    "catboost",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "adaboost",
    "neural_network",
    "logistic_regression",
    "svm"
]
