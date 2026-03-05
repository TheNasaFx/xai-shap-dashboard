"""
Tests for SHAP Explainer Module
================================

Unit tests for the SHAPExplainer class.

Author: XAI-SHAP Framework
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import xgboost, skip tests if not available
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None

# Try to import shap
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Skip all tests in this module if dependencies are missing
if not HAS_XGBOOST or not HAS_SHAP:
    pytest.skip("xgboost or shap not installed", allow_module_level=True)

from src.explainers.shap_explainer import SHAPExplainer


class TestSHAPExplainer:
    """Tests for SHAPExplainer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_classes=2,
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def trained_xgb_model(self, sample_data):
        """Create trained XGBoost model."""
        X, y = sample_data
        model = xgb.XGBClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X, y)
        return model
    
    @pytest.fixture
    def trained_rf_model(self, sample_data):
        """Create trained Random Forest model."""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model
    
    def test_explainer_initialization(self):
        """Test explainer initializes correctly."""
        explainer = SHAPExplainer()
        assert explainer is not None
    
    def test_fit_xgboost_model(self, trained_xgb_model, sample_data):
        """Test fitting explainer to XGBoost model."""
        X, _ = sample_data
        explainer = SHAPExplainer()
        explainer.fit(trained_xgb_model, X)
        
        assert explainer._explainer is not None
    
    def test_fit_random_forest_model(self, trained_rf_model, sample_data):
        """Test fitting explainer to Random Forest model."""
        X, _ = sample_data
        explainer = SHAPExplainer()
        explainer.fit(trained_rf_model, X)
        
        assert explainer._explainer is not None
    
    def test_compute_shap_values(self, trained_xgb_model, sample_data):
        """Test computing SHAP values."""
        X, _ = sample_data
        explainer = SHAPExplainer()
        explainer.fit(trained_xgb_model, X)
        
        shap_values = explainer.shap_values(X[:10])
        
        assert shap_values is not None
        assert shap_values.shape == (10, 5)  # 10 samples, 5 features
    
    def test_local_explanation(self, trained_xgb_model, sample_data):
        """Test local explanation for single instance."""
        X, _ = sample_data
        explainer = SHAPExplainer()
        explainer.fit(trained_xgb_model, X)
        
        explanation = explainer.explain_local(X[0])
        
        assert 'shap_values' in explanation
        assert 'feature_contributions' in explanation
    
    def test_global_explanation(self, trained_xgb_model, sample_data):
        """Test global explanation."""
        X, _ = sample_data
        explainer = SHAPExplainer()
        explainer.fit(trained_xgb_model, X)
        
        explanation = explainer.explain_global(X[:50])
        
        assert 'feature_importance' in explanation
        assert len(explanation['feature_importance']) == 5
    
    def test_shap_values_shape_consistency(self, trained_xgb_model, sample_data):
        """Test SHAP values shape matches input."""
        X, _ = sample_data
        explainer = SHAPExplainer()
        explainer.fit(trained_xgb_model, X)
        
        for n_samples in [1, 5, 20]:
            shap_values = explainer.shap_values(X[:n_samples])
            assert shap_values.shape[0] == n_samples
            assert shap_values.shape[1] == X.shape[1]
    
    def test_shap_values_not_all_zero(self, trained_xgb_model, sample_data):
        """Test that SHAP values are not all zero."""
        X, _ = sample_data
        explainer = SHAPExplainer()
        explainer.fit(trained_xgb_model, X)
        
        shap_values = explainer.shap_values(X[:10])
        
        assert not np.allclose(shap_values, 0)


class TestExplainerEdgeCases:
    """Test edge cases for explainer."""
    
    def test_explain_without_fit(self):
        """Test error when explaining without fitting."""
        explainer = SHAPExplainer()
        
        X = np.random.rand(10, 5)
        
        with pytest.raises((AttributeError, ValueError)):
            explainer.shap_values(X)
    
    def test_single_sample_explanation(self):
        """Test explaining single sample."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        model = xgb.XGBClassifier(n_estimators=10, eval_metric='logloss')
        model.fit(X, y)
        
        explainer = SHAPExplainer()
        explainer.fit(model, X)
        
        # Single sample as 1D array
        single_sample = X[0:1]
        shap_values = explainer.shap_values(single_sample)
        
        assert shap_values.shape == (1, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
