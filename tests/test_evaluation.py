"""
Tests for Evaluation Module
============================

Unit tests for model evaluation and fairness metrics.

Author: XAI-SHAP Framework
"""

import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import ModelEvaluator
from src.evaluation.fairness import FairnessEvaluator, BiasDetector


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance."""
        return ModelEvaluator()
    
    def test_classification_evaluation(self, evaluator):
        """Test classification metrics calculation."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 0, 1])
        
        results = evaluator.evaluate_classification(y_true, y_pred)
        
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1_score' in results
        assert 0 <= results['accuracy'] <= 1
    
    def test_perfect_classification(self, evaluator):
        """Test perfect classification scores."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        
        results = evaluator.evaluate_classification(y_true, y_pred)
        
        assert results['accuracy'] == 1.0
        assert results['precision'] == 1.0
        assert results['recall'] == 1.0
        assert results['f1_score'] == 1.0
    
    def test_regression_evaluation(self, evaluator):
        """Test regression metrics calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.0, 2.9, 4.2, 4.8])
        
        results = evaluator.evaluate_regression(y_true, y_pred)
        
        assert 'mse' in results
        assert 'rmse' in results
        assert 'mae' in results
        assert 'r2' in results
        assert results['mse'] >= 0
    
    def test_perfect_regression(self, evaluator):
        """Test perfect regression scores."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        results = evaluator.evaluate_regression(y_true, y_pred)
        
        assert results['mse'] == 0.0
        assert results['mae'] == 0.0
        assert results['r2'] == 1.0


class TestFairnessEvaluator:
    """Tests for FairnessEvaluator class."""
    
    @pytest.fixture
    def evaluator(self):
        """Create fairness evaluator instance."""
        return FairnessEvaluator()
    
    def test_demographic_parity(self, evaluator):
        """Test demographic parity calculation."""
        # Group 0: 1/3 positive, Group 1: 2/3 positive
        y_pred = np.array([1, 0, 0, 1, 1, 0])
        protected = np.array([0, 0, 0, 1, 1, 1])
        
        result = evaluator.demographic_parity(y_pred, protected)
        
        assert 'demographic_parity_difference' in result
        assert 'demographic_parity_ratio' in result
        assert 'group_0_positive_rate' in result
        assert 'group_1_positive_rate' in result
    
    def test_demographic_parity_perfect(self, evaluator):
        """Test perfect demographic parity."""
        # Both groups: 50% positive
        y_pred = np.array([1, 0, 1, 0])
        protected = np.array([0, 0, 1, 1])
        
        result = evaluator.demographic_parity(y_pred, protected)
        
        assert result['demographic_parity_difference'] == 0.0
        assert result['demographic_parity_ratio'] == 1.0
    
    def test_disparate_impact(self, evaluator):
        """Test disparate impact calculation."""
        y_pred = np.array([1, 0, 0, 1, 1, 1])
        protected = np.array([0, 0, 0, 1, 1, 1])
        
        result = evaluator.disparate_impact(y_pred, protected)
        
        assert 'disparate_impact_ratio' in result
        assert result['disparate_impact_ratio'] >= 0
    
    def test_equalized_odds(self, evaluator):
        """Test equalized odds calculation."""
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0])
        protected = np.array([0, 0, 0, 1, 1, 1])
        
        result = evaluator.equalized_odds(y_true, y_pred, protected)
        
        assert 'true_positive_rate_difference' in result
        assert 'false_positive_rate_difference' in result

    def test_evaluate_includes_group_performance_gaps(self, evaluator):
        """Test that fairness evaluation exposes per-group performance metrics and gap summaries."""

        class DummyBinaryModel:
            def __init__(self, predictions):
                self._predictions = np.array(predictions)

            def predict(self, X):
                return self._predictions[:len(X)]

        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0])
        protected = pd.DataFrame({'group': np.array([0, 0, 0, 1, 1, 1])})
        X_test = np.zeros((len(y_true), 2))

        result = evaluator.evaluate(
            model=DummyBinaryModel(y_pred),
            X_test=X_test,
            y_test=y_true,
            protected_attributes=['group'],
            protected_data=protected,
        )

        group_metrics = result['metrics_by_attribute']['group']['group_metrics']

        assert 'accuracy' in group_metrics['0']
        assert 'precision' in group_metrics['0']
        assert 'recall' in group_metrics['1']
        assert 'f1' in group_metrics['1']
        assert 'accuracy_gap' in result['metrics_by_attribute']['group']
        assert 'f1_gap' in result['metrics_by_attribute']['group']
        assert result['metrics_by_attribute']['group']['worst_group_by_f1'] in {'0', '1'}
        assert result['metrics_by_attribute']['group']['best_group_by_f1'] in {'0', '1'}


class TestBiasDetector:
    """Tests for BiasDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create bias detector instance."""
        return BiasDetector()
    
    def test_analyze_predictions(self, detector):
        """Test prediction analysis by group."""
        y_pred = np.array([1, 0, 0, 1, 1, 1])
        protected = np.array([0, 0, 0, 1, 1, 1])
        
        analysis = detector.analyze_predictions(y_pred, protected)
        
        assert 'group_statistics' in analysis
        assert 0 in analysis['group_statistics']
        assert 1 in analysis['group_statistics']
    
    def test_detect_bias_threshold(self, detector):
        """Test bias detection with threshold."""
        y_pred = np.array([0, 0, 0, 1, 1, 1])  # Group 0: 0%, Group 1: 100%
        protected = np.array([0, 0, 0, 1, 1, 1])
        
        has_bias = detector.has_significant_bias(
            y_pred, protected, threshold=0.1
        )
        
        assert has_bias == True


class TestFairnessEdgeCases:
    """Test edge cases for fairness evaluation."""
    
    def test_single_group(self):
        """Test with single group only."""
        evaluator = FairnessEvaluator()
        
        y_pred = np.array([1, 0, 1])
        protected = np.array([0, 0, 0])  # Only one group
        
        # Should handle gracefully (return 0 or handle empty group)
        result = evaluator.demographic_parity(y_pred, protected)
        
        # The function should still return a result
        assert result is not None
    
    def test_all_positive_predictions(self):
        """Test when all predictions are positive."""
        evaluator = FairnessEvaluator()
        
        y_pred = np.array([1, 1, 1, 1])
        protected = np.array([0, 0, 1, 1])
        
        result = evaluator.demographic_parity(y_pred, protected)
        
        assert result['group_0_positive_rate'] == 1.0
        assert result['group_1_positive_rate'] == 1.0
        assert result['demographic_parity_difference'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
