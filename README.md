# XAI-SHAP Visual Analytics Framework

## Framework Development of Visual Analytics of Black-Box Models with SHAP for Responsible and Explainable AI

**Хариуцлагатай, тайлбарлагдах боломжтой AI-д зориулсан SHAP ашиглан Black-Box загваруудын визуал аналитик хөгжүүлэх хүрээний боловсруулалт**

---

## 🎯 Project Overview

This framework provides a comprehensive solution for understanding and explaining black-box machine learning models using SHAP (SHapley Additive exPlanations). It combines:

- **Data Processing**: Bias detection, normalization, and preprocessing
- **Model Training**: Support for XGBoost, Random Forest, Neural Networks
- **SHAP Explanations**: Local and global feature importance analysis
- **Visual Analytics Dashboard**: Interactive, user-friendly explanations

## 📁 Project Structure

```
diploma/
├── docs/                          # Documentation & Thesis
│   ├── chapters/                  # Thesis chapters (Mongolian)
│   ├── figures/                   # Diagrams, screenshots
│   └── references/                # Bibliography & papers
├── src/                           # Source code
│   ├── core/                      # Core framework components
│   ├── data_processing/           # Data preprocessing module
│   ├── models/                    # ML models wrapper
│   ├── explainers/                # SHAP explanation engine
│   ├── visualization/             # Visual analytics components
│   ├── dashboard/                 # Streamlit dashboard app
│   ├── evaluation/                # Evaluation metrics & fairness
│   └── utils/                     # Utility functions
├── data/                          # Datasets
│   ├── raw/                       # Raw data files
│   ├── processed/                 # Processed data
│   └── sample/                    # Sample datasets for testing
├── notebooks/                     # Jupyter notebooks
│   ├── experiments/               # Experimental notebooks
│   └── tutorials/                 # Tutorial notebooks
├── tests/                         # Unit tests
├── config/                        # Configuration files
├── examples/                      # Example use cases
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
└── README.md                      # This file
```

## 🔬 Research Questions

1. How can we develop a system that provides understandable explanations to users?
2. Can SHAP be integrated with visual analytics to explain black-box model feature importance effectively?
3. How can we ensure fairness and transparency in AI model explanations?

## 🛠️ Technology Stack

- **Python 3.9+**
- **Machine Learning**: scikit-learn, XGBoost, TensorFlow/PyTorch
- **Explainability**: SHAP, LIME
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Dashboard**: Streamlit
- **Data Processing**: Pandas, NumPy

## 🚀 Quick Start

```bash
# Clone and setup
cd diploma
pip install -r requirements.txt

# Run the dashboard
streamlit run src/dashboard/app.py

# Run example
python examples/basic_usage.py
```

## 📊 Framework Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    XAI-SHAP Visual Analytics Framework          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │    Data     │  │   Model     │  │   SHAP      │  │ Visual  │ │
│  │ Processing  │→ │  Training   │→ │ Explainer   │→ │Analytics│ │
│  │   Module    │  │   Module    │  │   Module    │  │Dashboard│ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
│        ↓               ↓               ↓               ↓        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Responsible AI Evaluation Layer                ││
│  │         (Fairness Metrics, Transparency Checks)             ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## 📝 Thesis Chapters

1. **Introduction** - AI development, black-box models, XAI importance
2. **Literature Review** - XAI methods, visual analytics, Responsible AI
3. **System Design & Implementation** - Framework architecture and code
4. **Evaluation & Conclusion** - Testing, results, future work

## 👨‍💻 Author

Bachelor's Thesis in Computer Science - 2025/2026

## 📄 License

MIT License
