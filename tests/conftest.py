"""
Test Configuration
==================

Pytest configuration and fixtures.

Author: XAI-SHAP Framework
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


@pytest.fixture
def classification_data():
    """Generate sample classification data."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    return X, y


@pytest.fixture
def regression_data():
    """Generate sample regression data."""
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=5,
        noise=10,
        random_state=42
    )
    return X, y


@pytest.fixture
def sample_dataframe():
    """Generate sample DataFrame with target."""
    np.random.seed(42)
    n = 100
    
    df = pd.DataFrame({
        'age': np.random.randint(18, 70, n),
        'income': np.random.normal(50000, 15000, n),
        'score': np.random.randint(300, 850, n),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'target': np.random.choice([0, 1], n)
    })
    
    return df


@pytest.fixture
def fairness_data():
    """Generate data with protected attribute for fairness testing."""
    np.random.seed(42)
    n = 200
    
    protected = np.random.choice([0, 1], n, p=[0.5, 0.5])
    y_true = np.random.choice([0, 1], n)
    
    # Introduce bias: group 1 more likely to get positive prediction
    y_pred = np.where(
        protected == 1,
        np.random.choice([0, 1], n, p=[0.3, 0.7]),
        np.random.choice([0, 1], n, p=[0.6, 0.4])
    )
    
    return y_true, y_pred, protected
