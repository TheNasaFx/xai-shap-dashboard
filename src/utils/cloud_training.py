"""
Cloud Training - Клоуд сургалтын хэрэгсэл
==========================================

Том өгөгдлийн багцад зориулсан Google Colab notebook үүсгэгч.
Хэрэглэгч notebook-ийг татаж, Colab-д ажиллуулж, 
үнэгүй GPU/TPU ашиглан загвар сургах боломжтой.

Зохиогч: XAI-SHAP Framework
"""

import json
import base64
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Том өгөгдлийн босго (мөрийн тоо)
LARGE_DATASET_THRESHOLD = 10000


def is_large_dataset(n_rows: int) -> bool:
    """Өгөгдлийн багц том эсэхийг шалгах."""
    return n_rows >= LARGE_DATASET_THRESHOLD


def generate_colab_notebook(
    data_csv_base64: str,
    target_column: str,
    model_type: str,
    params: Dict[str, Any],
    protected_attributes: Optional[List[str]] = None,
    filename: str = "dataset.csv"
) -> dict:
    """
    Google Colab-д ажиллуулах .ipynb notebook үүсгэх.
    
    Параметрүүд:
        data_csv_base64: Өгөгдлийн CSV файлын base64 кодчилолт
        target_column: Зорилтот баганын нэр
        model_type: Загварын төрөл
        params: Hyperparameter-ууд
        protected_attributes: Хамгаалагдсан атрибутууд
        filename: Өгөгдлийн файлын нэр
    
    Буцаах:
        .ipynb notebook dictionary
    """
    
    params_str = json.dumps(params, indent=2, default=str)
    protected_str = json.dumps(protected_attributes or [], ensure_ascii=False)
    
    cells = [
        # Cell 1: Title
        _markdown_cell(f"""# XAI-SHAP Framework - Клоуд Сургалт
        
Энэ notebook нь **Google Colab**-ийн үнэгүй GPU/TPU нөөцийг ашиглан 
том өгөгдлийн багц дээр загвар сургах зорилготой.

**Загварын төрөл:** `{model_type}`  
**Зорилтот багана:** `{target_column}`  
**Өгөгдлийн файл:** `{filename}`

---
> **GPU идэвхжүүлэх:** Runtime → Change runtime type → T4 GPU сонгох
"""),
        
        # Cell 2: Install dependencies
        _code_cell("""# Шаардлагатай сангуудыг суулгах
!pip install -q xgboost lightgbm catboost shap scikit-learn pandas numpy plotly

import warnings
warnings.filterwarnings('ignore')

print("Сангууд амжилттай суулгагдлаа!")"""),
        
        # Cell 3: Load data from embedded base64
        _code_cell(f"""# Өгөгдлийг ачаалах (notebook-д суулгагдсан)
import base64
import io
import pandas as pd
import numpy as np

# Base64 кодчилолттой өгөгдлийг задлах
data_b64 = \"\"\"{data_csv_base64}\"\"\"

csv_bytes = base64.b64decode(data_b64)
df = pd.read_csv(io.BytesIO(csv_bytes))

print(f"Өгөгдлийн хэмжээ: {{df.shape[0]:,}} мөр, {{df.shape[1]}} багана")
print(f"Зорилтот багана: {target_column}")
print(f"\\nЭхний 5 мөр:")
df.head()"""),
        
        # Cell 4: Data preprocessing
        _code_cell(f"""# Өгөгдөл боловсруулах
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

target_column = "{target_column}"

# Шинж чанарууд болон зорилтыг тусгаарлах
X = df.drop(columns=[target_column])
y = df[target_column]

# Categorical баганнуудыг кодлох
for col in X.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Тоон баганнуудын дутуу утгыг бөглөх
X = X.fillna(X.median(numeric_only=True))

feature_names = X.columns.tolist()

# Сургалт/тест хуваалт
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.2, random_state=42
)

# Нормчлол
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Сургалтын дээжүүд: {{X_train.shape[0]:,}}")
print(f"Тест дээжүүд: {{X_test.shape[0]:,}}")  
print(f"Шинж чанарын тоо: {{X_train.shape[1]}}")"""),
        
        # Cell 5: Train model
        _code_cell(f"""# Загвар сургах
import time

model_type = "{model_type}"
params = {params_str}

print(f"{{model_type}} загвар сургаж байна...")
print(f"Hyperparameter-ууд: {{params}}")

start_time = time.time()

{_get_model_training_code(model_type)}

elapsed = time.time() - start_time
print(f"\\nСургалт дууслаа! ({{elapsed:.1f}} секунд)")"""),
        
        # Cell 6: Evaluate
        _code_cell("""# Загварын үнэлгээ
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    r2_score, mean_squared_error, mean_absolute_error
)

y_pred = model.predict(X_test)

# Даалгаврын төрлийг тодорхойлох
unique_values = np.unique(y_test)
is_classification = len(unique_values) <= 20

if is_classification:
    y_pred_rounded = np.round(y_pred).astype(int)
    
    print("=" * 50)
    print("ЗАГВАРЫН ГҮЙЦЭТГЭЛ (Classification)")
    print("=" * 50)
    print(f"Нарийвчлал (Accuracy):  {accuracy_score(y_test, y_pred_rounded):.4f}")
    print(f"Precision:              {precision_score(y_test, y_pred_rounded, average='weighted'):.4f}")
    print(f"Recall:                 {recall_score(y_test, y_pred_rounded, average='weighted'):.4f}")
    print(f"F1 Score:               {f1_score(y_test, y_pred_rounded, average='weighted'):.4f}")
    print(f"\\nДэлгэрэнгүй тайлан:")
    print(classification_report(y_test, y_pred_rounded))
else:
    print("=" * 50)
    print("ЗАГВАРЫН ГҮЙЦЭТГЭЛ (Regression)")
    print("=" * 50)
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE:     {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"MAE:      {mean_absolute_error(y_test, y_pred):.4f}")"""),
        
        # Cell 7: SHAP Explanations
        _code_cell("""# SHAP тайлбарууд үүсгэх
import shap

print("SHAP утгуудыг тооцоолж байна...")

# Explainer сонгох
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
except:
    explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
    shap_values = explainer.shap_values(X_test)

# Multi-output бол эхнийхийг авах
if isinstance(shap_values, list):
    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

print(f"SHAP утгууд тооцоологдлоо: {shap_values.shape}")"""),
        
        # Cell 8: Visualizations
        _code_cell("""# SHAP Визуализациуд
import plotly.graph_objects as go
import plotly.express as px

# 1. Шинж чанарын ач холбогдол (Bar plot)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
sorted_idx = np.argsort(mean_abs_shap)[::-1][:20]

fig = go.Figure(go.Bar(
    x=mean_abs_shap[sorted_idx],
    y=[feature_names[i] for i in sorted_idx],
    orientation='h',
    marker_color='#3b82f6'
))
fig.update_layout(
    title="Шинж Чанарын Ач Холбогдол (SHAP)",
    xaxis_title="Дундаж |SHAP утга|",
    yaxis=dict(autorange='reversed'),
    height=600
)
fig.show()

print("\\nТоп 10 Чухал Шинж Чанар:")
for i, idx in enumerate(sorted_idx[:10]):
    print(f"  {i+1}. {feature_names[idx]}: {mean_abs_shap[idx]:.4f}")"""),
        
        # Cell 9: Download results
        _code_cell("""# Үр дүнг хадгалах
import json

results = {
    'model_type': model_type,
    'feature_importance': [
        {'feature': feature_names[i], 'importance': float(mean_abs_shap[i])}
        for i in sorted_idx
    ],
    'shap_values_shape': list(shap_values.shape),
}

# JSON хадгалах
with open('xai_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Загварыг хадгалах
import joblib
joblib.dump(model, 'trained_model.pkl')

print("Файлууд хадгалагдлаа:")
print("  - xai_results.json (үр дүн)")
print("  - trained_model.pkl (загвар)")

# Google Colab-аас татахад
try:
    from google.colab import files
    files.download('xai_results.json')
    files.download('trained_model.pkl')
    print("\\nФайлууд автоматаар таны компьютерт татагдах болно.")
except:
    print("\\nColab биш орчинд ажиллаж байна. Файлуудыг гараар татна уу.")"""),
    ]
    
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": [],
                "name": "XAI-SHAP Cloud Training",
                "gpuType": "T4"
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "language_info": {
                "name": "python"
            },
            "accelerator": "GPU"
        },
        "cells": cells
    }
    
    return notebook


def _markdown_cell(source: str) -> dict:
    """Markdown cell үүсгэх."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def _code_cell(source: str) -> dict:
    """Code cell үүсгэх."""
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")],
        "outputs": [],
        "execution_count": None
    }


def _get_model_training_code(model_type: str) -> str:
    """Загварын сургалтын кодыг үүсгэх."""
    
    codes = {
        "xgboost": """import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

unique_vals = np.unique(y_train)
is_clf = len(unique_vals) <= 20

if is_clf:
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    xgb_params = {
        'n_estimators': params.get('n_estimators', 100),
        'max_depth': params.get('max_depth', 6),
        'learning_rate': params.get('learning_rate', 0.1),
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
        'tree_method': 'gpu_hist'  # GPU ашиглах
    }
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train_enc, eval_set=[(X_test, y_test_enc)], verbose=False)
else:
    xgb_params = {
        'n_estimators': params.get('n_estimators', 100),
        'max_depth': params.get('max_depth', 6),
        'learning_rate': params.get('learning_rate', 0.1),
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'gpu_hist'
    }
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)""",

        "lightgbm": """import lightgbm as lgb

unique_vals = np.unique(y_train)
is_clf = len(unique_vals) <= 20

lgb_params = {
    'n_estimators': params.get('n_estimators', 100),
    'max_depth': params.get('max_depth', 6),
    'learning_rate': params.get('learning_rate', 0.1),
    'random_state': 42,
    'n_jobs': -1,
    'device': 'gpu',  # GPU ашиглах
    'verbose': -1
}

if is_clf:
    model = lgb.LGBMClassifier(**lgb_params)
else:
    model = lgb.LGBMRegressor(**lgb_params)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)])""",

        "catboost": """from catboost import CatBoostClassifier, CatBoostRegressor

unique_vals = np.unique(y_train)
is_clf = len(unique_vals) <= 20

cb_params = {
    'iterations': params.get('n_estimators', 100),
    'depth': params.get('max_depth', 6),
    'learning_rate': params.get('learning_rate', 0.1),
    'random_seed': 42,
    'task_type': 'GPU',  # GPU ашиглах
    'verbose': 0
}

if is_clf:
    model = CatBoostClassifier(**cb_params)
else:
    model = CatBoostRegressor(**cb_params)

model.fit(X_train, y_train, eval_set=(X_test, y_test))""",

        "random_forest": """from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

unique_vals = np.unique(y_train)
is_clf = len(unique_vals) <= 20

rf_params = {
    'n_estimators': params.get('n_estimators', 100),
    'max_depth': params.get('max_depth', 10),
    'random_state': 42,
    'n_jobs': -1
}

if is_clf:
    model = RandomForestClassifier(**rf_params)
else:
    model = RandomForestRegressor(**rf_params)

model.fit(X_train, y_train)""",

        "neural_network": """from sklearn.neural_network import MLPClassifier, MLPRegressor

unique_vals = np.unique(y_train)
is_clf = len(unique_vals) <= 20

hidden = params.get('hidden_layers', [128, 64, 32])
if isinstance(hidden, list):
    hidden = tuple(hidden)

nn_params = {
    'hidden_layer_sizes': hidden,
    'max_iter': params.get('epochs', 200),
    'random_state': 42,
    'early_stopping': True,
    'verbose': False
}

if is_clf:
    model = MLPClassifier(**nn_params)
else:
    model = MLPRegressor(**nn_params)

model.fit(X_train, y_train)""",
    }
    
    # Default code for other model types
    default_code = """from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

unique_vals = np.unique(y_train)
is_clf = len(unique_vals) <= 20

rf_params = {
    'n_estimators': params.get('n_estimators', 100),
    'max_depth': params.get('max_depth', 10),
    'random_state': 42,
    'n_jobs': -1
}

if is_clf:
    model = RandomForestClassifier(**rf_params)
else:
    model = RandomForestRegressor(**rf_params)

model.fit(X_train, y_train)"""
    
    return codes.get(model_type, default_code)
