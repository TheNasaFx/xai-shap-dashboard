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


def __getattr__(name):
    if name == "ModelTrainer":
        from src.models.trainer import ModelTrainer
        return ModelTrainer
    elif name == "BaseModel":
        from src.models.base_model import BaseModel
        return BaseModel
    elif name == "RandomForestModel":
        from src.models.random_forest_model import RandomForestModel
        return RandomForestModel
    elif name == "XGBoostModel":
        try:
            from src.models.xgboost_model import XGBoostModel
            return XGBoostModel
        except ImportError:
            return None
    elif name == "NeuralNetworkModel":
        try:
            from src.models.neural_network_model import NeuralNetworkModel
            return NeuralNetworkModel
        except ImportError:
            return None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ModelTrainer",
    "BaseModel",
    "RandomForestModel",
    "XGBoostModel",
    "NeuralNetworkModel",
    "AVAILABLE_MODELS",
]
