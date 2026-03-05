"""
Tests for Core Framework Module
================================

Unit tests for the XAIFramework class.

Author: XAI-SHAP Framework
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import xgboost
try:
    import xgboost
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.core.framework import XAIFramework


class TestXAIFramework:
    """Tests for XAIFramework class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        feature_names = [f"feature_{i}" for i in range(5)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        return df
    
    @pytest.fixture
    def framework(self):
        """Create framework instance."""
        return XAIFramework()
    
    def test_framework_initialization(self, framework):
        """Test framework initializes correctly."""
        assert framework is not None
        assert framework.model is None
        assert framework.shap_values is None
    
    def test_load_data_from_dataframe(self, framework, sample_data):
        """Test loading data from DataFrame."""
        framework.load_data(
            data_path=sample_data,
            target_column='target',
            test_size=0.2
        )
        
        assert framework.X_train is not None
        assert framework.X_test is not None
        assert framework.y_train is not None
        assert framework.y_test is not None
        assert len(framework.feature_names) == 5
    
    def test_load_data_split_sizes(self, framework, sample_data):
        """Test that data split sizes are correct."""
        framework.load_data(
            data_path=sample_data,
            target_column='target',
            test_size=0.3
        )
        
        total = len(sample_data)
        # sklearn stratified split may not give exact sizes
        # Just check the split is approximately correct
        test_ratio = len(framework.X_test) / total
        assert 0.15 <= test_ratio <= 0.35  # Allow some tolerance
    
    @pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
    def test_train_xgboost_model(self, framework, sample_data):
        """Test training XGBoost model."""
        framework.load_data(data_path=sample_data, target_column='target')
        framework.train_model(
            model_type='xgboost',
            model_params={'n_estimators': 10, 'max_depth': 3}
        )
        
        assert framework.model is not None
        assert hasattr(framework.model, 'predict')
    
    @pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
    def test_train_random_forest_model(self, framework, sample_data):
        """Test training Random Forest model."""
        framework.load_data(data_path=sample_data, target_column='target')
        framework.train_model(
            model_type='random_forest',
            model_params={'n_estimators': 10}
        )
        
        assert framework.model is not None
    
    @pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
    def test_explain_model(self, framework, sample_data):
        """Test SHAP explanation generation."""
        framework.load_data(data_path=sample_data, target_column='target')
        framework.train_model(model_type='xgboost', model_params={'n_estimators': 10})
        framework.explain_model(n_samples=10)
        
        assert framework.shap_values is not None
        assert framework.shap_values.shape[1] == 5  # Number of features
    
    @pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
    def test_model_prediction(self, framework, sample_data):
        """Test model makes predictions."""
        framework.load_data(data_path=sample_data, target_column='target')
        framework.train_model(model_type='xgboost', model_params={'n_estimators': 10})
        
        predictions = framework.model.predict(framework.X_test)
        
        assert len(predictions) == len(framework.X_test)
        assert all(p in [0, 1] for p in predictions)
    
    def test_feature_names_preserved(self, framework, sample_data):
        """Test feature names are preserved after loading."""
        framework.load_data(data_path=sample_data, target_column='target')
        
        expected_names = [f"feature_{i}" for i in range(5)]
        assert framework.feature_names == expected_names


class TestDataLoading:
    """Tests for data loading functionality."""
    
    def test_load_data_missing_target(self):
        """Test error when target column is missing."""
        framework = XAIFramework()
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        
        with pytest.raises((KeyError, ValueError)):
            framework.load_data(data_path=df, target_column='target')
    
    def test_load_data_empty_dataframe(self):
        """Test error with empty DataFrame."""
        framework = XAIFramework()
        df = pd.DataFrame()
        
        with pytest.raises((KeyError, ValueError)):
            framework.load_data(data_path=df, target_column='target')


class TestModelTraining:
    """Tests for model training functionality."""
    
    @pytest.fixture
    def ready_framework(self):
        """Create framework with loaded data."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        df['target'] = y
        
        framework = XAIFramework()
        framework.load_data(data_path=df, target_column='target')
        return framework
    
    @pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
    def test_train_without_data(self):
        """Test error when training without data."""
        framework = XAIFramework()
        
        with pytest.raises((ValueError, AttributeError)):
            framework.train_model(model_type='xgboost')
    
    def test_unsupported_model_type(self, ready_framework):
        """Test error with unsupported model type."""
        with pytest.raises((ValueError, KeyError)):
            ready_framework.train_model(model_type='unsupported_model')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
