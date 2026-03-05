# Test Suite

This directory contains unit tests for the XAI-SHAP Visual Analytics Framework.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_framework.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Test Files

- **test_framework.py**: Tests for core XAIFramework class
- **test_explainer.py**: Tests for SHAP explainer functionality
- **test_evaluation.py**: Tests for model evaluation and fairness metrics
- **conftest.py**: Pytest fixtures and configuration

## Writing Tests

Follow pytest conventions:

- Test files should start with `test_`
- Test functions should start with `test_`
- Use fixtures for common setup

Example:

```python
import pytest
from src.core.framework import XAIFramework

def test_framework_initialization():
    framework = XAIFramework()
    assert framework is not None
```
