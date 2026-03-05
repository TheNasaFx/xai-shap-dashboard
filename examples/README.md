# Examples Directory

# ==================

This directory contains example scripts demonstrating how to use the XAI-SHAP Visual Analytics Framework.

## Available Examples

### 1. basic_usage.py

Complete walkthrough of the framework's core functionality:

- Data loading and preprocessing
- Model training (XGBoost)
- SHAP explanation generation
- Visualization creation
- Report generation

```bash
python examples/basic_usage.py
```

### 2. pipeline_example.py

Demonstrates the automated pipeline for end-to-end analysis:

- Configuration-based workflow
- Automated stage execution
- Results collection

```bash
python examples/pipeline_example.py
```

### 3. dashboard_example.py

Interactive dashboard launcher:

- Streamlit-based UI
- Real-time analysis
- Interactive visualizations

```bash
python examples/dashboard_example.py
```

### 4. fairness_example.py

Responsible AI features demonstration:

- Bias detection in data
- Fairness metrics evaluation
- SHAP-based fairness analysis

```bash
python examples/fairness_example.py
```

## Quick Start

```python
from src.core.framework import XAIFramework

# Initialize
framework = XAIFramework()

# Load data
framework.load_data(
    data_path="data/sample/breast_cancer.csv",
    target_column="diagnosis"
)

# Train model
framework.train_model(model_type="xgboost")

# Explain
framework.explain_model()

# Visualize
framework.visualize(plot_type="summary", save_path="outputs/summary.html")
```

## Output

All examples save their outputs to the `outputs/` directory:

- HTML visualizations
- Analysis reports
- JSON data files
