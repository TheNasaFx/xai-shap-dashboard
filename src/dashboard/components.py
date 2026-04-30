"""
Dashboard Components — Deep Dark Minimalist UI
===============================================
Apple-inspired clean design with dark theme.
"""

import html
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.dashboard.state import (
    clear_model_state,
    get_workflow_status,
    invalidate_after_data_load,
    invalidate_after_model_change,
    replace_uploaded_data,
    reset_dashboard_session,
    sync_dashboard_state,
)
from src.utils.helpers import get_binary_probability_scores, predict_with_threshold

logger = logging.getLogger(__name__)

# ============================================================================
# Icon System
# ============================================================================
ICONS = {
    "check": "✓", "pending": "○", "data": "⬡", "model": "⬢",
    "explain": "◇", "chart": "▣", "fairness": "◎", "settings": "⚙",
    "arrow": "→", "bullet": "·", "refresh": "↻", "upload": "↑",
    "download": "↓", "play": "▶", "target": "◉", "info": "ℹ",
    "warning": "!", "up": "▲", "down": "▼",
}

# ============================================================================
# Deep Dark Theme CSS
# ============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary: #0a0a0a;
    --bg-secondary: #111111;
    --bg-card: #161616;
    --bg-card-hover: #1a1a1a;
    --bg-elevated: #1e1e1e;
    --border: #222222;
    --border-subtle: #1a1a1a;
    --text-primary: #f5f5f5;
    --text-secondary: #b0b0b0;
    --text-muted: #777777;
    --accent: #3b82f6;
    --accent-hover: #60a5fa;
    --success: #22c55e;
    --warning: #eab308;
    --error: #ef4444;
    --gradient-accent: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
}

/* === Global === */
.main { padding: 0 1rem; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
.block-container { max-width: 1200px; padding: 2rem 1rem; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* === Section Header === */
.sec-head {
    font-size: 1.5rem; font-weight: 700; color: var(--text-primary);
    margin: 0 0 1.5rem 0; padding: 0 0 0.75rem 0;
    border-bottom: 1px solid var(--border);
    letter-spacing: -0.03em;
}

/* === Cards === */
.card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.25rem; margin: 0.5rem 0;
    transition: all 0.2s ease;
}
.card:hover { background: var(--bg-card-hover); border-color: #333; }

.focus-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.02) 0%, rgba(59,130,246,0.03) 100%);
    border: 1px solid var(--border); border-radius: 14px;
    padding: 1rem 1.1rem; margin: 0.35rem 0 0.75rem 0;
    min-height: 178px;
}
.focus-kicker {
    color: var(--text-muted); font-size: 0.72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.55rem;
}
.focus-title {
    color: var(--text-primary); font-size: 1rem; font-weight: 700;
    letter-spacing: -0.02em; margin-bottom: 0.45rem;
}
.focus-body {
    color: var(--text-secondary); font-size: 0.88rem; line-height: 1.55;
}
.focus-footer {
    margin-top: 0.8rem; padding-top: 0.7rem; border-top: 1px solid rgba(255,255,255,0.06);
    color: #d5d5d5; font-size: 0.8rem; line-height: 1.45;
}
.pill {
    display: inline-block; border-radius: 999px; padding: 0.18rem 0.55rem;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase; margin-bottom: 0.75rem;
}
.pill-ok { background: rgba(34,197,94,0.12); color: #7ef0a4; }
.pill-warn { background: rgba(234,179,8,0.12); color: #f9d86b; }
.pill-info { background: rgba(59,130,246,0.12); color: #9cc2ff; }

.intro-strip {
    display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap;
    margin: -0.35rem 0 1rem 0; padding: 0.65rem 0.85rem;
    border: 1px solid rgba(255,255,255,0.06); border-radius: 10px;
    background: rgba(255,255,255,0.02);
}
.intro-label {
    color: var(--text-primary); font-size: 0.7rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em;
}
.intro-text {
    color: var(--text-secondary); font-size: 0.82rem; line-height: 1.45;
}

.smart-callout {
    border-radius: 12px; padding: 0.95rem 1rem; margin: 0.75rem 0;
    border: 1px solid rgba(255,255,255,0.06); background: rgba(255,255,255,0.02);
}
.smart-callout.info { border-color: rgba(59,130,246,0.2); }
.smart-callout.ok { border-color: rgba(34,197,94,0.2); }
.smart-callout.warn { border-color: rgba(234,179,8,0.2); }
.smart-head {
    display: flex; align-items: center; justify-content: space-between; gap: 0.75rem;
    margin-bottom: 0.8rem;
}
.smart-title {
    color: var(--text-primary); font-size: 0.95rem; font-weight: 700;
    letter-spacing: -0.02em;
}
.smart-count {
    color: var(--text-muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;
}
.smart-item {
    display: grid; grid-template-columns: auto 1fr; gap: 0.75rem; align-items: start;
    padding: 0.48rem 0; border-top: 1px solid rgba(255,255,255,0.05);
}
.smart-item:first-of-type { border-top: none; padding-top: 0; }
.smart-tag {
    min-width: 52px; text-align: center; border-radius: 999px; padding: 0.18rem 0.5rem;
    font-size: 0.66rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase;
    background: rgba(59,130,246,0.12); color: #9cc2ff;
}
.smart-tag.warn { background: rgba(234,179,8,0.12); color: #f9d86b; }
.smart-tag.ok { background: rgba(34,197,94,0.12); color: #7ef0a4; }
.smart-tag.neutral { background: rgba(255,255,255,0.08); color: #d8d8d8; }
.smart-text {
    color: var(--text-secondary); font-size: 0.86rem; line-height: 1.5;
}
.sidebar-note {
    margin: 0.45rem 0 0.75rem 0; color: var(--text-secondary); font-size: 0.78rem; line-height: 1.45;
}

/* === Metric Cards === */
.m-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.5rem; text-align: center;
}
.m-val { font-size: 2rem; font-weight: 800; color: var(--text-primary); line-height: 1.2; letter-spacing: -0.04em; }
.m-lbl { font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 500; }

/* === Alert Boxes === */
.a-info {
    background: rgba(59,130,246,0.06); border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0; padding: 1rem 1.25rem; margin: 0.75rem 0;
    font-size: 0.875rem; color: #c8d6e5; line-height: 1.6;
}
.a-ok {
    background: rgba(34,197,94,0.06); border-left: 3px solid var(--success);
    border-radius: 0 8px 8px 0; padding: 1rem 1.25rem; margin: 0.75rem 0;
    color: var(--success); font-size: 0.875rem;
}
.a-warn {
    background: rgba(234,179,8,0.06); border-left: 3px solid var(--warning);
    border-radius: 0 8px 8px 0; padding: 1rem 1.25rem; margin: 0.75rem 0;
    color: var(--warning); font-size: 0.875rem;
}
.a-err {
    background: rgba(239,68,68,0.06); border-left: 3px solid var(--error);
    border-radius: 0 8px 8px 0; padding: 1rem 1.25rem; margin: 0.75rem 0;
    color: var(--error); font-size: 0.875rem;
}

/* === Status Indicator === */
.st-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
.st-on { background: var(--success); box-shadow: 0 0 6px var(--success); }
.st-off { background: var(--text-muted); }

/* === Sidebar === */
[data-testid="stSidebar"] { background: var(--bg-secondary) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif !important; }

/* === Buttons === */
.stButton > button {
    border-radius: 8px; font-weight: 600; font-family: 'Inter', sans-serif;
    transition: all 0.15s ease; letter-spacing: -0.01em;
}
.stButton > button:hover { transform: translateY(-1px); }
.stButton > button[kind="primary"] { background: var(--gradient-accent) !important; border: none; }

/* === Charts === */
.stPlotlyChart { width: 100% !important; border-radius: 12px; overflow: hidden; }
.js-plotly-plot .plotly { border-radius: 12px; }

/* === Tabs === */
.stTabs [data-baseweb="tab-list"] { gap: 2px; background: var(--bg-secondary); padding: 4px; border-radius: 10px; border: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { padding: 8px 16px; border-radius: 8px; font-weight: 500; font-size: 0.85rem; }
.stTabs [aria-selected="true"] { background: var(--bg-elevated) !important; }

/* === DataFrames === */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* === Metrics Override === */
[data-testid="metric-container"] {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px; padding: 0.875rem;
}
[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; }

/* === Expander === */
.streamlit-expanderHeader { font-weight: 500; font-size: 0.9rem; }

/* === Responsive === */
@media (max-width: 768px) {
    .m-val { font-size: 1.5rem; }
    .sec-head { font-size: 1.25rem; }
    .block-container { padding: 1rem 0.5rem; }
}

/* === Scrollbar === */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #444; }
</style>
"""

# ============================================================================
# Plotly Dark Theme
# ============================================================================
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,22,22,0.8)",
    font=dict(family="Inter, sans-serif", size=12, color="#888"),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    colorway=["#3b82f6", "#22c55e", "#eab308", "#ef4444", "#8b5cf6", "#06b6d4", "#f97316", "#ec4899"],
)


def _plotly_chart(st_module, fig, **kwargs):
    """Responsive plotly chart helper."""
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(autosize=True)
    try:
        st_module.plotly_chart(fig, width="stretch", config={"responsive": True, "displayModeBar": False})
    except Exception:
        st_module.plotly_chart(fig, width="stretch")


def inject_custom_css():
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# Helper Utils
# ============================================================================
def _fmt_dtype(dtype) -> str:
    m = {'int64': 'int', 'int32': 'int', 'float64': 'float', 'float32': 'float',
         'object': 'text', 'bool': 'bool', 'datetime64[ns]': 'datetime', 'category': 'category'}
    return m.get(str(dtype), str(dtype))

def _fmt_val(v) -> str:
    if isinstance(v, (np.integer,)): return str(int(v))
    if isinstance(v, (np.floating,)): return f"{float(v):.4f}"
    return str(v)

def _need(st_module, msg="Өмнөх алхмыг гүйцэтгэнэ үү."):
    st_module.markdown(f'<div class="a-warn">{msg}</div>', unsafe_allow_html=True)


def _friendly_error_message(error: Exception) -> str:
    raw_message = str(error).strip() or type(error).__name__
    lowered = raw_message.lower()

    if 'shape' in lowered or 'feature' in lowered:
        return 'Өгөгдөл болон идэвхтэй загварын feature бүтэц зөрж байна. Dataset-ээ дахин боловсруулаад model-оо шинээр сургана уу.'
    if 'could not convert' in lowered or 'dtype' in lowered or 'string to float' in lowered:
        return 'Өгөгдлийн төрөл тохирохгүй байна. Numeric биш багана, missing value, эсвэл encoding шаардлагатай талбар байгаа эсэхийг шалгана уу.'
    if 'predict_proba' in lowered or 'probability' in lowered:
        return 'Энэ үйлдэл probability score шаарддаг. Probability дэмждэг model сонгох эсвэл өөр metric/view ашиглана уу.'
    if 'shap' in lowered or 'callable' in lowered:
        return 'SHAP тайлбар үүсгэх үед алдаа гарлаа. Active model, feature бүтэц, мөн dataset хэмжээ тохиромжтой эсэхийг шалгана уу.'
    if 'memory' in lowered:
        return 'Санах ойн нөөц хүрэлцэхгүй байна. Dataset хэмжээг багасгах эсвэл хөнгөн analysis сонгоно уу.'
    if 'protected' in lowered or 'fairness' in lowered:
        return 'Fairness шинжилгээнд protected attribute мэдээлэл дутуу байна. Өгөгдөл боловсруулах үед protected attributes-аа дахин тохируулна уу.'

    return raw_message


def _handle_action_error(st_module, context: str, error: Exception):
    user_message = _friendly_error_message(error)
    logger.exception("%s failed: %s", context, error)
    st_module.session_state['last_error'] = {
        'context': context,
        'message': user_message,
        'detail': str(error).strip() or type(error).__name__,
    }
    st_module.markdown(
        f'''<div class="a-err"><strong>{html.escape(context)}</strong><br/>{html.escape(user_message)}
        <div style="margin-top:0.4rem;color:#f5c2c7;font-size:0.75rem;opacity:0.85;">Technical detail: {html.escape(st_module.session_state['last_error']['detail'])}</div></div>''',
        unsafe_allow_html=True,
    )


def _section_intro(st_module, title: str, message: str):
    st_module.markdown(
        f'<div class="intro-strip"><span class="intro-label">{html.escape(title)}</span>'
        f'<span class="intro-text">{html.escape(message)}</span></div>',
        unsafe_allow_html=True,
    )

def _check_framework(st_module):
    return sync_dashboard_state(st_module)


def _safe_target_preview(analyzer, data, target_col):
    preview_method = getattr(analyzer, 'analyze_target_column', None)
    if callable(preview_method):
        return preview_method(data, target_col)

    fallback_method = getattr(analyzer, '_analyze_target_column', None)
    if callable(fallback_method):
        return fallback_method(data, target_col)

    report = analyzer.analyze(data, target=target_col)
    return report.get('target_analysis') or {
        'task_type': 'unknown',
        'unique_count': 0,
        'missing_percent': 0.0,
        'recommended_metric': '—',
        'issues': [],
    }


def _compute_model_metrics(model, X, y):
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
    )

    predictions = model.predict(X)
    unique_values = np.unique(y)
    is_classification = len(unique_values) <= 20 and np.all(unique_values == unique_values.astype(int))

    if is_classification:
        rounded_predictions = predictions.round().astype(int)
        metrics = {
            'accuracy': accuracy_score(y, rounded_predictions),
            'f1': f1_score(y, rounded_predictions, average='weighted', zero_division=0),
            'precision': precision_score(y, rounded_predictions, average='weighted', zero_division=0),
            'recall': recall_score(y, rounded_predictions, average='weighted', zero_division=0),
        }
        if len(unique_values) == 2 and hasattr(model, 'predict_proba'):
            try:
                metrics['auc'] = roc_auc_score(y, model.predict_proba(X)[:, 1])
            except Exception:
                pass
        metrics['primary_metric'] = metrics['f1']
        metrics['primary_metric_name'] = 'f1'
        return metrics, True

    metrics = {
        'r2': r2_score(y, predictions),
        'rmse': np.sqrt(mean_squared_error(y, predictions)),
        'mae': mean_absolute_error(y, predictions),
    }
    metrics['primary_metric'] = metrics['r2']
    metrics['primary_metric_name'] = 'r2'
    return metrics, False


def _compute_cross_validation_summary(config, model_type, params, X, y, cv_folds):
    from sklearn.model_selection import KFold, StratifiedKFold
    from src.models.trainer import ModelTrainer

    y_array = np.asarray(y)
    unique_values = np.unique(y_array)
    is_classification = len(unique_values) <= 20 and np.all(unique_values == unique_values.astype(int))

    if is_classification:
        class_counts = pd.Series(y_array).value_counts()
        effective_folds = min(int(cv_folds), int(class_counts.min()))
        if effective_folds < 2:
            return None
        splitter = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=42)
        split_iterator = splitter.split(X, y_array)
    else:
        effective_folds = min(int(cv_folds), len(X))
        if effective_folds < 2:
            return None
        splitter = KFold(n_splits=effective_folds, shuffle=True, random_state=42)
        split_iterator = splitter.split(X)

    scores = []
    for train_idx, valid_idx in split_iterator:
        trainer = ModelTrainer(config)
        model = trainer.train(
            model_type=model_type,
            X_train=X[train_idx],
            y_train=y_array[train_idx],
            X_test=X[valid_idx],
            y_test=y_array[valid_idx],
            **params,
        )
        fold_metrics, _ = _compute_model_metrics(model, X[valid_idx], y_array[valid_idx])
        scores.append(float(fold_metrics['primary_metric']))

    metric_name = 'f1' if is_classification else 'r2'
    return {
        'metric_name': metric_name,
        'folds': effective_folds,
        'scores': scores,
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
    }


def _find_best_classification_threshold(y_true, y_probabilities):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    best_result = None
    for threshold in np.linspace(0.1, 0.9, 17):
        predictions = (y_probabilities >= threshold).astype(int)
        result = {
            'threshold': float(threshold),
            'accuracy': float(accuracy_score(y_true, predictions)),
            'f1': float(f1_score(y_true, predictions, zero_division=0)),
            'precision': float(precision_score(y_true, predictions, zero_division=0)),
            'recall': float(recall_score(y_true, predictions, zero_division=0)),
        }
        if best_result is None or result['f1'] > best_result['f1']:
            best_result = result

    return best_result


def _get_active_shap_values(framework, explanations):
    shap_values = framework.shap_values if framework is not None and framework.shap_values is not None else explanations.get('shap_values')
    if shap_values is None:
        return None
    return np.asarray(shap_values)


def _resolve_base_value(base_value, sample_index=0):
    if base_value is None:
        return None

    base_array = np.asarray(base_value)
    if base_array.ndim == 0:
        return float(base_array)
    if base_array.ndim == 1:
        safe_index = min(sample_index, len(base_array) - 1)
        return float(base_array[safe_index])

    safe_index = min(sample_index, base_array.shape[0] - 1)
    if base_array.shape[1] > 0:
        return float(base_array[safe_index, 0])
    return float(base_array[safe_index].mean())


def _get_sample_prediction_context(framework, sample_index, decision_threshold=None):
    sample_index = min(sample_index, len(framework.X_test) - 1)
    sample_array = framework.X_test[sample_index:sample_index + 1]
    predicted_value = predict_with_threshold(framework.model, sample_array, decision_threshold)[0]
    predicted_probability = None

    if hasattr(framework.model, 'predict_proba'):
        try:
            probability_scores = get_binary_probability_scores(framework.model, sample_array)
            if probability_scores is not None:
                predicted_probability = float(np.ravel(probability_scores)[0])
        except Exception:
            predicted_probability = None

    actual_value = None
    if framework.y_test is not None and sample_index < len(framework.y_test):
        actual_value = framework.y_test[sample_index]

    return {
        'sample_index': sample_index,
        'predicted_value': predicted_value,
        'predicted_probability': predicted_probability,
        'actual_value': actual_value,
    }


def _build_local_compare_rows(shap_values, feature_names, sample_index_a, sample_index_b, top_n):
    safe_a = min(sample_index_a, len(shap_values) - 1)
    safe_b = min(sample_index_b, len(shap_values) - 1)
    sample_a = shap_values[safe_a]
    sample_b = shap_values[safe_b]
    delta = sample_b - sample_a
    sorted_indices = np.argsort(np.abs(delta))[::-1][:top_n]

    rows = []
    for feature_index in sorted_indices:
        rows.append({
            'Feature': feature_names[feature_index],
            'Sample A SHAP': float(sample_a[feature_index]),
            'Sample B SHAP': float(sample_b[feature_index]),
            'Delta (B-A)': float(delta[feature_index]),
            'Magnitude': float(abs(delta[feature_index])),
        })

    return rows


def _build_subgroup_shap_rows(framework, shap_values, feature_names, attribute_name, max_groups=8):
    protected_test_data = getattr(framework, '_protected_test_data', None)
    if protected_test_data is None or attribute_name not in protected_test_data.columns:
        return []
    if len(protected_test_data) != len(shap_values):
        return []

    group_series = protected_test_data[attribute_name].fillna('Missing').astype(str)
    group_counts = group_series.value_counts().head(max_groups)
    prediction_vector = np.asarray(framework.model.predict(framework.X_test))

    rows = []
    for group_name in group_counts.index:
        group_indices = np.flatnonzero(group_series.to_numpy() == group_name)
        if len(group_indices) == 0:
            continue

        group_shap = shap_values[group_indices]
        mean_abs_shap = np.abs(group_shap).mean(axis=0)
        top_feature_index = int(np.argmax(mean_abs_shap))
        observed_mean = framework.y_test[group_indices].mean() if framework.y_test is not None else None

        rows.append({
            'Group': group_name,
            'Samples': int(len(group_indices)),
            'Mean prediction': float(np.mean(prediction_vector[group_indices])),
            'Observed mean': float(observed_mean) if observed_mean is not None else None,
            'Top feature': feature_names[top_feature_index],
            'Top |SHAP|': float(mean_abs_shap[top_feature_index]),
        })

    return rows


def _build_group_metric_rows(group_metrics):
    rows = []
    for group_name, metrics in group_metrics.items():
        rows.append({
            'Бүлэг': str(group_name),
            'Хэмжээ': metrics.get('size', 0),
            'Эерэг %': float(metrics.get('positive_rate', 0)),
            'Accuracy': float(metrics.get('accuracy', 0)),
            'Precision': float(metrics.get('precision', 0)),
            'Recall': float(metrics.get('recall', 0)),
            'F1': float(metrics.get('f1', 0)),
            'TPR': float(metrics.get('true_positive_rate', 0)) if metrics.get('true_positive_rate') is not None else None,
            'FPR': float(metrics.get('false_positive_rate', 0)) if metrics.get('false_positive_rate') is not None else None,
        })

    return rows


def _format_group_metric_display(group_rows):
    group_df = pd.DataFrame(group_rows)
    if group_df.empty:
        return group_df, group_df

    display_df = group_df.copy()
    for column in ['Эерэг %', 'Accuracy', 'Precision', 'Recall', 'F1', 'TPR', 'FPR']:
        display_df[column] = display_df[column].map(lambda value: '—' if pd.isna(value) else f'{float(value):.2%}')

    return group_df, display_df


def _render_focus_card(st_module, kicker, title, body, tone='info', footer=None, badge=None):
    tone_class = {
        'ok': 'pill-ok',
        'warn': 'pill-warn',
        'info': 'pill-info',
    }.get(tone, 'pill-info')
    badge_html = f'<span class="pill {tone_class}">{html.escape(badge)}</span>' if badge else ''
    footer_html = f'<div class="focus-footer">{html.escape(footer)}</div>' if footer else ''
    st_module.markdown(
        f'<div class="focus-card">{badge_html}<div class="focus-kicker">{html.escape(kicker)}</div>'
        f'<div class="focus-title">{html.escape(title)}</div>'
        f'<div class="focus-body">{html.escape(body)}</div>{footer_html}</div>',
        unsafe_allow_html=True,
    )


def _parse_smart_item(text):
    cleaned = str(text).strip()
    match = re.match(r'^\[([^\]]+)\]\s*(.+)$', cleaned)
    if not match:
        return None, cleaned
    return match.group(1).strip().upper(), match.group(2).strip()


def _smart_tag_tone(tag, default_tone='info'):
    if tag in {'WARN', 'RISK', 'ALERT'}:
        return 'warn'
    if tag in {'SAFE', 'OK', 'PASS'}:
        return 'ok'
    if tag in {'KEY', 'DATA', 'LINK', 'TARGET', 'TIP', 'STEP'}:
        return 'info'
    return 'neutral' if default_tone == 'info' else default_tone


def _render_bullet_callout(st_module, title, items, tone='info'):
    filtered = [str(item).strip() for item in items if str(item).strip()]
    if not filtered:
        return

    rows_html = []
    for index, item in enumerate(filtered, start=1):
        tag, body = _parse_smart_item(item)
        tag_label = tag or f'{index:02d}'
        tag_tone = _smart_tag_tone(tag, tone)
        rows_html.append(
            f'<div class="smart-item"><span class="smart-tag {tag_tone}">{html.escape(tag_label)}</span>'
            f'<div class="smart-text">{html.escape(body)}</div></div>'
        )

    st_module.markdown(
        f'<div class="smart-callout {tone}"><div class="smart-head"><div class="smart-title">{html.escape(title)}</div>'
        f'<div class="smart-count">{len(filtered)} items</div></div>{"".join(rows_html)}</div>',
        unsafe_allow_html=True,
    )


def _clean_summary_lines(summary_text, strip_tags=False):
    clean = str(summary_text or '').replace('=', '').strip()
    lines = []
    seen = set()
    for raw_line in clean.splitlines():
        line = raw_line.strip().lstrip('-').strip()
        if not line:
            continue
        if line.startswith('[') and line.endswith(':'):
            continue
        if 'ТАЙЛАН' in line.upper() and len(line) <= 40:
            continue
        if strip_tags:
            _, line = _parse_smart_item(line)
        if not line or line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return lines


def _get_advanced_analysis_specs():
    return {
        'Автомат Дүгнэлт': {
            'kicker': 'Executive readout',
            'question': 'Яг одоо ямар feature-үүд prediction-ийг удирдаж байна вэ?',
            'output': 'Key findings, threshold clues, interactions, risk and protective factors.',
            'next_step': 'SHAP тайлбараа thesis defense дээр яаж тайлбарлах вэ гэдгийн эхний хариуг эндээс авна.',
        },
        'Тогтвортой Байдал': {
            'kicker': 'Reliability check',
            'question': 'Энэ тайлбарууд тогтвортой, дахин давтагдахуйц байна уу?',
            'output': 'Overall stability score, unstable features, ranking consistency.',
            'next_step': 'Тогтворгүй feature-үүд өндөр байвал conclusion-оо болгоомжтой гаргана.',
        },
        'Counterfactual': {
            'kicker': 'Action pathway',
            'question': 'Ямар өөрчлөлт хийвэл prediction өөрчлөгдөх вэ?',
            'output': 'Minimal feature changes and direction-aware recommendations.',
            'next_step': 'Case-level recommendation эсвэл what-if analysis-д ашиглана.',
        },
        'Алдааны Шинжилгээ': {
            'kicker': 'Failure diagnosis',
            'question': 'Model хаана, ямар pattern дээр алдаж байна вэ?',
            'output': 'Error mix, severity patterns, feature correlations, case review.',
            'next_step': 'Data augmentation, threshold tuning, subgroup mitigation-д шууд ашиглана.',
        },
    }


def _get_analysis_status_meta(analysis_name, analysis_results):
    if analysis_name == 'Counterfactual':
        return 'Interactive', 'info'
    if analysis_name in analysis_results:
        return 'Ready', 'ok'
    return 'Not run', 'warn'


def _classify_fairness_risk(metrics):
    f1_gap = float(metrics.get('f1_gap', 0.0))
    accuracy_gap = float(metrics.get('accuracy_gap', 0.0))
    dp_ratio = float(metrics.get('demographic_parity_ratio', 1.0))
    disparity_score = max(f1_gap, accuracy_gap, abs(1 - dp_ratio))

    if (not metrics.get('is_fair', True)) or disparity_score >= 0.25:
        return 'Өндөр эрсдэл', 'warn'
    if disparity_score >= 0.12:
        return 'Анхаарах', 'info'
    return 'Тогтвортой', 'ok'


def _build_fairness_audit_rows(metrics_by_attribute):
    rows = []
    for attribute_name, metrics in metrics_by_attribute.items():
        risk_label, _ = _classify_fairness_risk(metrics)
        rows.append({
            'Attribute': attribute_name,
            'Status': 'Pass' if metrics.get('is_fair') else 'Review',
            'Risk': risk_label,
            'DP Ratio': float(metrics.get('demographic_parity_ratio', 0)),
            'Disparate Impact': float(metrics.get('disparate_impact', 0)),
            'Accuracy Gap': float(metrics.get('accuracy_gap', 0)),
            'F1 Gap': float(metrics.get('f1_gap', 0)),
            'Worst Group': str(metrics.get('worst_group_by_f1') or '—'),
            'Best Group': str(metrics.get('best_group_by_f1') or '—'),
        })

    return sorted(rows, key=lambda row: (row['Status'] == 'Pass', row['F1 Gap']))


def _normalize_report_value(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_normalize_report_value(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return [_normalize_report_value(item) for item in value.to_dict(orient='records')]
    if isinstance(value, pd.Series):
        return {str(key): _normalize_report_value(item) for key, item in value.to_dict().items()}
    if isinstance(value, dict):
        return {str(key): _normalize_report_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_report_value(item) for item in value]
    if hasattr(value, 'to_dict') and callable(value.to_dict):
        return _normalize_report_value(value.to_dict())
    if hasattr(value, '__dict__'):
        return _normalize_report_value(vars(value))
    return str(value)


def _collect_recommendation_lines(recommendations, limit=8):
    lines = []
    for item in recommendations or []:
        if isinstance(item, dict):
            message = str(item.get('message', '')).strip()
            action = str(item.get('action', '')).strip()
            priority = str(item.get('priority', '')).strip()
            combined = ' | '.join(part for part in [priority.upper() if priority else '', message, action] if part)
            if combined:
                lines.append(combined)
        else:
            cleaned = str(item).strip()
            if cleaned:
                lines.append(cleaned)
    return lines[:limit]


def _count_structural_warnings(structural):
    if not structural:
        return 0

    explicit_counts = [
        int(value)
        for key, value in structural.items()
        if key.endswith('_count') and isinstance(value, (int, float, np.integer, np.floating))
    ]
    if explicit_counts:
        return int(sum(max(0, value) for value in explicit_counts))

    total = 0
    for value in structural.values():
        if isinstance(value, (list, tuple, set, dict, pd.Series, np.ndarray)):
            total += len(value)

    return int(total)


def _build_quality_report_summary(report):
    if not report:
        return {'available': False}

    target_analysis = report.get('target_analysis') or {}
    leakage = report.get('leakage_risks', {}) or {}
    structural = report.get('structural_risks', {}) or {}
    raw_quality_score = report.get('quality_score', 0)
    quality_score = raw_quality_score.get('score', 0) if isinstance(raw_quality_score, dict) else raw_quality_score

    return {
        'available': True,
        'quality_score': float(quality_score or 0),
        'missing_percent': float(report.get('missing_values', {}).get('total_missing_percent', 0) or 0),
        'duplicate_rows': int(report.get('duplicates', {}).get('duplicate_rows', 0) or 0),
        'leakage_risk_count': int(leakage.get('risk_count', 0) or 0),
        'high_risk_count': int(leakage.get('high_risk_count', 0) or 0),
        'target_profile': {
            'target': target_analysis.get('target'),
            'task_type': target_analysis.get('task_type'),
            'recommended_metric': target_analysis.get('recommended_metric'),
            'unique_count': target_analysis.get('unique_count'),
            'missing_percent': target_analysis.get('missing_percent'),
            'issue_count': len(target_analysis.get('issues', [])),
        },
        'structural_warning_count': _count_structural_warnings(structural),
        'top_leakage_features': [item.get('feature') for item in leakage.get('details', [])[:5]],
        'recommendations': _collect_recommendation_lines(report.get('recommendations', []), limit=6),
    }


def _build_model_report(trained_models, active_model_name):
    rows = []
    for name, details in (trained_models or {}).items():
        rows.append({
            'name': name,
            'active': name == active_model_name,
            'train_time': float(details.get('train_time', 0) or 0),
            'metrics': _normalize_report_value(details.get('metrics', {})),
            'train_metrics': _normalize_report_value(details.get('train_metrics', {})),
            'cv_summary': _normalize_report_value(details.get('cv_summary', {})),
            'overfit_gap': float(details.get('overfit_gap', 0) or 0),
            'params': _normalize_report_value(details.get('params', {})),
        })
    return rows


def _build_shap_report(fw, explanations):
    shap_values = getattr(fw, 'shap_values', None)
    if shap_values is None and explanations:
        shap_values = explanations.get('shap_values')
    feature_names = (explanations or {}).get('feature_names') or getattr(fw, 'feature_names', None) or []

    if shap_values is None or len(feature_names) == 0:
        return {'available': False}

    mean_abs_shap = np.abs(np.asarray(shap_values)).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:10]
    subgroup_summaries = {}
    protected_test_data = getattr(fw, '_protected_test_data', None)
    if protected_test_data is not None:
        for attribute in protected_test_data.columns[:3]:
            subgroup_summaries[attribute] = _build_subgroup_shap_rows(fw, np.asarray(shap_values), feature_names, attribute, max_groups=5)

    return {
        'available': True,
        'explained_rows': int(np.asarray(shap_values).shape[0]),
        'explained_features': int(np.asarray(shap_values).shape[1]) if np.asarray(shap_values).ndim >= 2 else len(feature_names),
        'top_features': [
            {
                'feature': feature_names[index],
                'importance': float(mean_abs_shap[index]),
            }
            for index in top_indices
        ],
        'global_summary': (explanations or {}).get('global', {}).get('summary'),
        'protected_attributes': list(getattr(fw, '_protected_attributes', []) or []),
        'subgroup_summaries': subgroup_summaries,
    }


def _build_analysis_report(analysis_results):
    if not analysis_results:
        return {'available': False}

    insights = analysis_results.get('Автомат Дүгнэлт') or {}
    stability = analysis_results.get('Тогтвортой Байдал') or {}
    error_report = analysis_results.get('Алдааны Шинжилгээ')
    counterfactual = analysis_results.get('Counterfactual') or {}

    error_summary = None
    if error_report is not None:
        error_summary = {
            'total_errors': int(getattr(error_report, 'total_errors', 0)),
            'false_positives': int(getattr(error_report, 'false_positives', 0)),
            'false_negatives': int(getattr(error_report, 'false_negatives', 0)),
            'top_patterns': [
                {
                    'description': pattern.description,
                    'severity': pattern.severity,
                    'affected_count': pattern.affected_count,
                }
                for pattern in getattr(error_report, 'patterns', [])[:5]
            ],
            'recommendations': _collect_recommendation_lines(getattr(error_report, 'recommendations', []), limit=6),
        }

    return {
        'available': True,
        'insights': {
            'key_findings': _normalize_report_value(insights.get('key_findings', [])),
            'summary': _clean_summary_lines(insights.get('summary', ''), strip_tags=True),
            'top_feature': _normalize_report_value((insights.get('feature_importance') or {}).get('top_feature')),
            'threshold_count': int((insights.get('thresholds') or {}).get('count', 0) or 0),
            'strongest_interaction': _normalize_report_value((insights.get('interactions') or {}).get('strongest_interaction')),
            'risk_factors': _normalize_report_value((insights.get('risk_factors') or [])[:5]),
            'protective_factors': _normalize_report_value((insights.get('protective_factors') or [])[:5]),
        } if insights else None,
        'stability': {
            'overall_score': float(stability.get('overall_stability_score', 0) or 0),
            'method': stability.get('method'),
            'valid_iterations': stability.get('valid_iterations', stability.get('n_iterations')),
            'summary': _clean_summary_lines(stability.get('summary', ''), strip_tags=True),
        } if stability else None,
        'error_analysis': error_summary,
        'counterfactual': _normalize_report_value(counterfactual) if counterfactual else None,
    }


def _build_fairness_report(fairness_results):
    if not fairness_results or 'metrics_by_attribute' not in fairness_results:
        return {'available': False}

    return {
        'available': True,
        'overall_fairness': bool(fairness_results.get('overall_fairness', False)),
        'audit_rows': _normalize_report_value(_build_fairness_audit_rows(fairness_results.get('metrics_by_attribute', {}))),
        'metrics_by_attribute': _normalize_report_value(fairness_results.get('metrics_by_attribute', {})),
        'recommendations': _collect_recommendation_lines(fairness_results.get('recommendations', []), limit=8),
    }


def _build_dashboard_report_data(st_module, fw):
    workflow = get_workflow_status(st_module)
    selected_model_name = st_module.session_state.get('selected_model_name') or (type(fw.model).__name__ if fw and fw.model is not None else None)
    trained_models = st_module.session_state.get('trained_models', {}) or {}
    explanations = st_module.session_state.get('explanations', {}) or {}
    processing_info = getattr(fw, 'processing_info', None) or st_module.session_state.get('processing_info') or {}
    preprocessing_config = st_module.session_state.get('preprocessing_config', {}) or {}

    report_data = {
        'title': 'XAI-SHAP Dashboard Report',
        'generated_at': datetime.now().isoformat(),
        'overview': {
            'dataset_name': workflow.get('dataset_name') or 'Unknown dataset',
            'target_column': workflow.get('target_col'),
            'active_model': selected_model_name,
            'decision_threshold': float(st_module.session_state.get('decision_threshold', 0.5)),
            'row_count': int(workflow.get('row_count', 0)),
            'column_count': int(workflow.get('column_count', 0)),
            'train_samples': int(len(fw.X_train)) if fw and getattr(fw, 'X_train', None) is not None else 0,
            'test_samples': int(len(fw.X_test)) if fw and getattr(fw, 'X_test', None) is not None else 0,
            'feature_count': int(len(getattr(fw, 'feature_names', []) or [])),
            'protected_attributes': list(getattr(fw, '_protected_attributes', []) or []),
        },
        'workflow': {
            'data_loaded': bool(workflow.get('data_loaded')),
            'model_trained': bool(workflow.get('model_trained')),
            'explanations_generated': bool(workflow.get('explanations_generated')),
            'completed_steps': int(workflow.get('completed_steps', 0)),
        },
        'preprocessing': {
            'ui_config': _normalize_report_value(preprocessing_config),
            'processing_info': _normalize_report_value(processing_info),
        },
        'data_quality': _build_quality_report_summary(st_module.session_state.get('quality_report')),
        'models': _build_model_report(trained_models, selected_model_name),
        'performance': {
            'active_model': selected_model_name,
            'active_threshold': float(st_module.session_state.get('decision_threshold', 0.5)),
            'evaluation_results': _normalize_report_value(getattr(fw, '_evaluation_results', {})),
        },
        'explanations': _build_shap_report(fw, explanations),
        'analysis': _build_analysis_report(st_module.session_state.get('advanced_analysis_results', {})),
        'fairness': _build_fairness_report(st_module.session_state.get('fairness_results')),
        'last_error': _normalize_report_value(st_module.session_state.get('last_error')),
    }

    sections = []
    for label, key in [
        ('Data quality', 'data_quality'),
        ('Models', 'models'),
        ('Explanations', 'explanations'),
        ('Analysis', 'analysis'),
        ('Fairness', 'fairness'),
    ]:
        section_value = report_data[key]
        if isinstance(section_value, list):
            ready = len(section_value) > 0
            item_count = len(section_value)
        else:
            ready = bool(section_value and section_value.get('available', True))
            item_count = 0
        sections.append({'Section': label, 'Ready': 'Yes' if ready else 'No', 'Items': item_count})
    report_data['section_coverage'] = sections
    return report_data


def _build_markdown_report(data):
    lines = [
        f"# {data['title']}",
        f"Generated: {data['generated_at']}",
        '',
        '## Overview',
        f"- Dataset: {data['overview'].get('dataset_name', '—')}",
        f"- Target: {data['overview'].get('target_column', '—')}",
        f"- Active model: {data['overview'].get('active_model', '—')}",
        f"- Threshold: {data['overview'].get('decision_threshold', 0.5):.2f}",
        f"- Rows: {data['overview'].get('row_count', 0)} | Columns: {data['overview'].get('column_count', 0)}",
        '',
        '## Workflow',
        f"- Data loaded: {data['workflow'].get('data_loaded')}",
        f"- Model trained: {data['workflow'].get('model_trained')}",
        f"- Explanations ready: {data['workflow'].get('explanations_generated')}",
        '',
        '## Data Quality',
    ]

    quality = data.get('data_quality', {})
    if quality.get('available'):
        lines.extend([
            f"- Quality score: {quality.get('quality_score', 0):.1f}",
            f"- Missing percent: {quality.get('missing_percent', 0):.1f}%",
            f"- Duplicate rows: {quality.get('duplicate_rows', 0)}",
            f"- Leakage risks: {quality.get('leakage_risk_count', 0)}",
        ])
        for rec in quality.get('recommendations', []):
            lines.append(f"- {rec}")
    else:
        lines.append('- Not generated')

    lines.extend(['', '## Models'])
    if data.get('models'):
        for model in data['models']:
            metric_parts = [f"{key}={value:.4f}" for key, value in (model.get('metrics') or {}).items() if isinstance(value, (int, float))]
            lines.append(f"- {model['name']}{' (active)' if model.get('active') else ''}: {', '.join(metric_parts)}")
    else:
        lines.append('- No model comparison data')

    lines.extend(['', '## Explanations'])
    explanations = data.get('explanations', {})
    if explanations.get('available'):
        for item in explanations.get('top_features', [])[:10]:
            lines.append(f"- {item['feature']}: {item['importance']:.4f}")
    else:
        lines.append('- No SHAP explanations available')

    lines.extend(['', '## Analysis'])
    analysis = data.get('analysis', {})
    insights = analysis.get('insights') or {}
    if insights:
        for item in insights.get('key_findings', [])[:6]:
            lines.append(f"- {re.sub(r'^\[[^\]]+\]\s*', '', str(item))}")
    stability = analysis.get('stability') or {}
    if stability:
        lines.append(f"- Stability score: {stability.get('overall_score', 0):.2f}")
    error_summary = analysis.get('error_analysis') or {}
    if error_summary:
        lines.append(f"- Total errors: {error_summary.get('total_errors', 0)}")

    lines.extend(['', '## Fairness'])
    fairness = data.get('fairness', {})
    if fairness.get('available'):
        lines.append(f"- Overall fairness: {fairness.get('overall_fairness')}")
        for row in fairness.get('audit_rows', []):
            lines.append(
                f"- {row['Attribute']}: status={row['Status']}, risk={row['Risk']}, DP ratio={row['DP Ratio']:.3f}, F1 gap={row['F1 Gap']:.3f}"
            )
    else:
        lines.append('- Fairness audit not generated')

    return '\n'.join(lines)


def _build_html_report(data):
    def _kv_table(rows):
        body = ''.join(
            f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
            for key, value in rows
        )
        return f'<table class="report-table">{body}</table>'

    def _rows_table(headers, rows):
        if not rows:
            return '<p class="empty">No data available.</p>'
        head_html = ''.join(f'<th>{html.escape(str(header))}</th>' for header in headers)
        body_html = ''
        for row in rows:
            body_html += '<tr>' + ''.join(f'<td>{html.escape(str(row.get(header, "—")))}</td>' for header in headers) + '</tr>'
        return f'<table class="report-table"><thead><tr>{head_html}</tr></thead><tbody>{body_html}</tbody></table>'

    model_rows = []
    for model in data.get('models', []):
        metric_summary = ', '.join(
            f"{key}={value:.4f}" for key, value in (model.get('metrics') or {}).items() if isinstance(value, (int, float))
        )
        model_rows.append({
            'Model': f"{model['name']}{' (active)' if model.get('active') else ''}",
            'Metrics': metric_summary or '—',
            'CV': (model.get('cv_summary') or {}).get('mean', '—'),
            'Overfit gap': f"{model.get('overfit_gap', 0):.4f}",
            'Train time': f"{model.get('train_time', 0):.2f}s",
        })

    top_feature_rows = [
        {'Feature': item['feature'], 'Importance': f"{item['importance']:.4f}"}
        for item in data.get('explanations', {}).get('top_features', [])
    ]

    fairness_rows = []
    for row in data.get('fairness', {}).get('audit_rows', []):
        fairness_rows.append({
            'Attribute': row['Attribute'],
            'Status': row['Status'],
            'Risk': row['Risk'],
            'DP Ratio': f"{row['DP Ratio']:.3f}",
            'F1 Gap': f"{row['F1 Gap']:.3f}",
            'Worst Group': row['Worst Group'],
        })

    insight_items = ''.join(
        f'<li>{html.escape(re.sub(r"^\[[^\]]+\]\s*", "", str(item)))}</li>'
        for item in (data.get('analysis', {}).get('insights') or {}).get('key_findings', [])[:6]
    )
    quality_recommendations = ''.join(
        f'<li>{html.escape(str(item))}</li>'
        for item in data.get('data_quality', {}).get('recommendations', [])[:6]
    )
    fairness_recommendations = ''.join(
        f'<li>{html.escape(str(item))}</li>'
        for item in data.get('fairness', {}).get('recommendations', [])[:6]
    )

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(data['title'])}</title>
  <style>
    :root {{ --ink:#132238; --muted:#5d6b82; --line:#d7deea; --paper:#f4f7fb; --card:#ffffff; --blue:#2563eb; --green:#15803d; --amber:#b45309; --red:#b91c1c; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:linear-gradient(180deg,#eef4ff 0%,#f8fbff 240px,#f4f7fb 240px); color:var(--ink); font-family:'Aptos','Segoe UI',sans-serif; }}
    .page {{ max-width:1120px; margin:0 auto; padding:32px 20px 60px; }}
    .hero {{ background:linear-gradient(135deg,#0f172a 0%,#1d4ed8 100%); color:#fff; border-radius:24px; padding:28px 30px; box-shadow:0 20px 45px rgba(15,23,42,0.18); }}
    .hero h1 {{ margin:0 0 10px; font-size:2rem; letter-spacing:-0.04em; }}
    .hero p {{ margin:0; color:rgba(255,255,255,0.82); }}
    .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; margin:22px 0; }}
    .kpi {{ background:rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.14); border-radius:16px; padding:14px 16px; }}
    .kpi-value {{ font-size:1.35rem; font-weight:800; letter-spacing:-0.04em; }}
    .kpi-label {{ font-size:0.76rem; text-transform:uppercase; letter-spacing:0.08em; color:rgba(255,255,255,0.74); margin-top:6px; }}
    .section {{ background:var(--card); border:1px solid var(--line); border-radius:20px; padding:22px; margin-top:18px; box-shadow:0 12px 25px rgba(15,23,42,0.05); }}
    .section h2 {{ margin:0 0 14px; font-size:1.18rem; letter-spacing:-0.03em; }}
    .muted {{ color:var(--muted); }}
    .report-table {{ width:100%; border-collapse:collapse; margin-top:10px; }}
    .report-table th, .report-table td {{ border-bottom:1px solid var(--line); padding:10px 8px; text-align:left; vertical-align:top; font-size:0.92rem; }}
    .report-table th {{ color:var(--muted); font-size:0.74rem; text-transform:uppercase; letter-spacing:0.08em; }}
    .tag {{ display:inline-block; border-radius:999px; padding:4px 10px; font-size:0.72rem; font-weight:700; letter-spacing:0.06em; text-transform:uppercase; }}
    .tag.ok {{ background:#dcfce7; color:var(--green); }}
    .tag.warn {{ background:#fef3c7; color:var(--amber); }}
    ul {{ margin:10px 0 0 18px; padding:0; }}
    li {{ margin:6px 0; line-height:1.5; }}
    .empty {{ color:var(--muted); margin:0; }}
  </style>
</head>
<body>
  <div class=\"page\">
    <section class=\"hero\">
      <h1>{html.escape(data['title'])}</h1>
      <p>Generated at {html.escape(data['generated_at'][:19].replace('T', ' '))}</p>
      <div class=\"kpis\">
        <div class=\"kpi\"><div class=\"kpi-value\">{html.escape(str(data['overview'].get('dataset_name', '—')))}</div><div class=\"kpi-label\">Dataset</div></div>
        <div class=\"kpi\"><div class=\"kpi-value\">{html.escape(str(data['overview'].get('active_model', '—')))}</div><div class=\"kpi-label\">Active model</div></div>
        <div class=\"kpi\"><div class=\"kpi-value\">{data['overview'].get('decision_threshold', 0.5):.2f}</div><div class=\"kpi-label\">Threshold</div></div>
        <div class=\"kpi\"><div class=\"kpi-value\">{data['overview'].get('feature_count', 0)}</div><div class=\"kpi-label\">Features</div></div>
      </div>
    </section>
    <section class=\"section\"><h2>Overview</h2>{_kv_table([
        ('Target', data['overview'].get('target_column', '—')),
        ('Rows', data['overview'].get('row_count', 0)),
        ('Columns', data['overview'].get('column_count', 0)),
        ('Train samples', data['overview'].get('train_samples', 0)),
        ('Test samples', data['overview'].get('test_samples', 0)),
        ('Protected attributes', ', '.join(data['overview'].get('protected_attributes', [])) or '—'),
    ])}</section>
    <section class=\"section\"><h2>Data Quality</h2>{_kv_table([
        ('Quality score', f"{data.get('data_quality', {}).get('quality_score', 0):.1f}"),
        ('Missing percent', f"{data.get('data_quality', {}).get('missing_percent', 0):.1f}%"),
        ('Duplicate rows', data.get('data_quality', {}).get('duplicate_rows', 0)),
        ('Leakage risks', data.get('data_quality', {}).get('leakage_risk_count', 0)),
        ('Target metric', (data.get('data_quality', {}).get('target_profile') or {}).get('recommended_metric', '—')),
    ])}<ul>{quality_recommendations}</ul></section>
    <section class=\"section\"><h2>Models</h2>{_rows_table(['Model','Metrics','CV','Overfit gap','Train time'], model_rows)}</section>
    <section class=\"section\"><h2>SHAP Explanation Summary</h2>{_rows_table(['Feature','Importance'], top_feature_rows)}</section>
    <section class=\"section\"><h2>Analysis Highlights</h2><ul>{insight_items}</ul>{_kv_table([
        ('Stability score', f"{(data.get('analysis', {}).get('stability') or {}).get('overall_score', 0):.2f}"),
        ('Threshold clues', (data.get('analysis', {}).get('insights') or {}).get('threshold_count', 0)),
        ('Total errors', (data.get('analysis', {}).get('error_analysis') or {}).get('total_errors', 0)),
    ])}</section>
    <section class=\"section\"><h2>Fairness Audit</h2><div class=\"muted\">Overall status: <span class=\"tag {'ok' if data.get('fairness', {}).get('overall_fairness') else 'warn'}\">{'Pass' if data.get('fairness', {}).get('overall_fairness') else 'Review'}</span></div>{_rows_table(['Attribute','Status','Risk','DP Ratio','F1 Gap','Worst Group'], fairness_rows)}<ul>{fairness_recommendations}</ul></section>
  </div>
</body>
</html>"""


def _build_report_download_payload(report_data, fmt):
    fmt_lower = str(fmt).lower()
    timestamp = report_data['generated_at'][:19].replace(':', '-').replace('T', '_')
    if fmt_lower == 'json':
        return json.dumps(report_data, ensure_ascii=False, indent=2), f'xai_report_{timestamp}.json', 'application/json'
    if fmt_lower == 'markdown':
        return _build_markdown_report(report_data), f'xai_report_{timestamp}.md', 'text/markdown'
    return _build_html_report(report_data), f'xai_report_{timestamp}.html', 'text/html'


# ============================================================================
# Sidebar
# ============================================================================
def render_sidebar():
    import streamlit as st

    fw = _check_framework(st)
    dl = st.session_state.get('data_loaded', False)
    mt = st.session_state.get('model_trained', False)
    sg = st.session_state.get('explanations_generated', False)

    st.markdown("### Статус")
    for label, active in [("Өгөгдөл", dl), ("Загвар", mt), ("SHAP", sg)]:
        dot = "st-on" if active else "st-off"
        txt = "Бэлэн" if active else "—"
        st.markdown(f'<span class="st-dot {dot}"></span> **{label}** {txt}', unsafe_allow_html=True)

    last_error = st.session_state.get('last_error')
    if last_error:
        st.markdown("### Сүүлийн Алдаа")
        st.markdown(
            f'<div class="a-err"><strong>{html.escape(last_error["context"])}</strong><br/>{html.escape(last_error["message"])}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Report export
    st.markdown("### Тайлан")
    fmt = st.selectbox("Формат", ["HTML", "Markdown", "JSON"], key="exp_fmt", label_visibility="collapsed")
    _export_report(st, fw, fmt)

    st.divider()

    if st.button(f"{ICONS['refresh']} Дахин Тохируулах", width="stretch"):
        reset_dashboard_session(st)
        st.rerun()

    st.markdown("""
    <div style="margin-top:2rem; font-size:0.7rem; color:#555;">
        XAI-SHAP Framework v2.0<br/>Дипломын ажил · 2026
    </div>
    """, unsafe_allow_html=True)


def _export_report(st_module, fw, fmt):
    """Тайлан экспорт UI ба comprehensive payload үүсгэх."""
    if fw is None or fw.model is None:
        st_module.caption("Тайлан идэвхжүүлэхийн тулд эхлээд загвар сургана уу.")
        return

    report_data = _build_dashboard_report_data(st_module, fw)
    payload, file_name, mime = _build_report_download_payload(report_data, fmt)
    ready_sections = sum(int(row['Ready'] == 'Yes') for row in report_data.get('section_coverage', []))
    st_module.markdown(
        f'<div class="sidebar-note"><strong>Coverage</strong><br/>{ready_sections}/{len(report_data.get("section_coverage", []))} section тайланд орно.</div>',
        unsafe_allow_html=True,
    )
    with st_module.expander('Тайланд багтах хэсгүүд'):
        st_module.dataframe(pd.DataFrame(report_data.get('section_coverage', [])), width='stretch', hide_index=True)
    st_module.download_button(
        f"{ICONS['download']} Тайлан Татах",
        payload,
        file_name,
        mime,
        width='stretch',
    )


# ============================================================================
# 1. DATA SECTION
# ============================================================================
def render_data_section():
    import streamlit as st
    from src.data_processing.data_quality import DataQualityAnalyzer
    fw = _check_framework(st)

    st.markdown('<div class="sec-head">Өгөгдөл</div>', unsafe_allow_html=True)
    _section_intro(st, "Data Intake", "Upload, target, protected attributes, preprocessing.")

    # Upload row
    c1, c2 = st.columns([3, 1])
    with c1:
        uploaded = st.file_uploader("CSV файл", type=['csv'], label_visibility="collapsed")
    with c2:
        use_sample = st.button("Жишээ өгөгдөл", width="stretch")

    if use_sample:
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer()
        df = pd.DataFrame(d.data, columns=d.feature_names)
        df['diagnosis'] = d.target
        replace_uploaded_data(
            st,
            df,
            dataset_name="Breast Cancer Sample",
            uploaded_file_signature="sample:breast_cancer",
        )
        st.rerun()

    if uploaded is not None:
        uploaded_signature = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get('uploaded_file_signature') != uploaded_signature:
            df = pd.read_csv(uploaded)
            replace_uploaded_data(
                st,
                df,
                dataset_name=uploaded.name,
                uploaded_file_signature=uploaded_signature,
            )
            st.rerun()

    if st.session_state.get('uploaded_data') is None:
        st.markdown('<div class="a-info">CSV файл байршуулах эсвэл жишээ өгөгдөл ашиглана уу.</div>', unsafe_allow_html=True)
        return

    df = st.session_state['uploaded_data']

    # Quick stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Мөр", f"{len(df):,}")
    c2.metric("Багана", df.shape[1])
    c3.metric("Дутуу", df.isnull().sum().sum())
    c4.metric("Тоон", len(df.select_dtypes(include=[np.number]).columns))

    # Preview
    with st.expander("Урьдчилсан харагдац", expanded=True):
        st.dataframe(df.head(8), width="stretch")

    # Column info
    with st.expander("Баганын мэдээлэл"):
        ci = pd.DataFrame({
            'Багана': df.columns,
            'Төрөл': [_fmt_dtype(d) for d in df.dtypes],
            'Өвөрмөц': df.nunique().values,
            'Дутуу %': [f"{100*df[c].isnull().sum()/len(df):.1f}" for c in df.columns]
        })
        st.dataframe(ci, width="stretch", hide_index=True)

    # Target selection
    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    with c1:
        target_col = st.selectbox("Зорилтот багана", df.columns.tolist(), index=len(df.columns)-1)
    with c2:
        protected = st.multiselect("Хамгаалагдсан атрибут", [c for c in df.columns if c != target_col])

    preprocessing_defaults = st.session_state.get('preprocessing_config', {})
    with st.expander("Preprocessing strategy", expanded=True):
        p1, p2 = st.columns(2)
        missing_strategy = p1.selectbox(
            "Missing value strategy",
            ["median", "mean", "mode", "drop"],
            index=["median", "mean", "mode", "drop"].index(preprocessing_defaults.get('missing_strategy', 'median')),
        )
        encoding_method = p2.selectbox(
            "Categorical encoding",
            ["onehot", "label"],
            index=["onehot", "label"].index(preprocessing_defaults.get('encoding_method', 'onehot')),
        )
        p3, p4 = st.columns(2)
        normalization_method = p3.selectbox(
            "Normalization",
            ["standard", "minmax", "robust", "none"],
            index=["standard", "minmax", "robust", "none"].index(preprocessing_defaults.get('normalization_method', 'standard')),
        )
        test_size = p4.slider(
            "Test split",
            min_value=0.1,
            max_value=0.4,
            value=float(preprocessing_defaults.get('test_size', 0.2)),
            step=0.05,
        )

    current_preprocessing = {
        'missing_strategy': missing_strategy,
        'encoding_method': encoding_method,
        'normalization_method': normalization_method,
        'test_size': test_size,
    }
    st.session_state['preprocessing_config'] = current_preprocessing

    target_preview = _safe_target_preview(DataQualityAnalyzer(), df, target_col)
    preview_cols = st.columns(4)
    preview_cols[0].metric("Target төрөл", target_preview.get('task_type', 'unknown'))
    preview_cols[1].metric("Unique", target_preview.get('unique_count', 0))
    preview_cols[2].metric("Missing %", f"{target_preview.get('missing_percent', 0):.1f}")
    preview_cols[3].metric("Metric", target_preview.get('recommended_metric', '—'))

    preview_issues = target_preview.get('issues', [])
    if preview_issues:
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        top_issue = sorted(preview_issues, key=lambda item: severity_order.get(item['severity'], 3))[0]
        box_class = 'a-err' if top_issue['severity'] == 'high' else 'a-warn'
        st.markdown(
            f'<div class="{box_class}"><strong>Target suitability warning</strong><br/>{html.escape(top_issue["message"])}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="a-ok">Target column анхан шатны шалгалтаар тохиромжтой харагдаж байна.</div>', unsafe_allow_html=True)

    split_train_count = max(1, int(round(len(df) * (1 - test_size))))
    split_test_count = max(1, len(df) - split_train_count)
    split_preview_cols = st.columns([1, 1])

    with split_preview_cols[0]:
        st.markdown("#### Split Preview")
        st.dataframe(
            pd.DataFrame([
                {'Subset': 'Train', 'Rows': split_train_count, 'Percent': f'{(1 - test_size) * 100:.0f}%'},
                {'Subset': 'Test', 'Rows': split_test_count, 'Percent': f'{test_size * 100:.0f}%'},
            ]),
            width="stretch",
            hide_index=True,
        )
        if split_test_count < 30:
            st.markdown('<div class="a-warn">Test subset маш жижиг байна. Metric тогтворгүй болж магадгүй.</div>', unsafe_allow_html=True)

    with split_preview_cols[1]:
        st.markdown("#### Target Distribution")
        try:
            import plotly.graph_objects as go

            if target_preview.get('task_type') == 'classification':
                value_counts = df[target_col].fillna('Missing').astype(str).value_counts().head(12)
                fig = go.Figure(
                    data=[go.Bar(
                        x=value_counts.index.tolist(),
                        y=value_counts.values.tolist(),
                        marker_color='#3b82f6',
                        text=value_counts.values.tolist(),
                        textposition='auto',
                    )]
                )
                fig.update_layout(title_text='Class counts', height=280, margin=dict(l=20, r=20, t=40, b=20))
            else:
                numeric_target = pd.to_numeric(df[target_col], errors='coerce').dropna()
                fig = go.Figure(
                    data=[go.Histogram(
                        x=numeric_target,
                        marker_color='#22c55e',
                        nbinsx=min(30, max(10, int(np.sqrt(max(len(numeric_target), 1))))),
                    )]
                )
                fig.update_layout(title_text='Target histogram', height=280, margin=dict(l=20, r=20, t=40, b=20))

            _plotly_chart(st, fig)
        except Exception as e:
            _handle_action_error(st, 'Target distribution preview', e)

    if st.button(f"{ICONS['play']} Боловсруулах", type="primary", width="stretch"):
        with st.spinner("Боловсруулж байна..."):
            try:
                fw.config.set('data_processing.missing_value_strategy', missing_strategy)
                fw.config.set('data_processing.categorical_encoding', encoding_method)
                fw.config.set('data_processing.normalization_method', normalization_method)
                fw.config.set('data_processing.test_size', test_size)

                fw.data_processor.missing_strategy = missing_strategy
                fw.data_processor.encoding_method = encoding_method
                fw.data_processor.normalization_method = normalization_method
                fw.data_processor.test_size = test_size

                fw.load_data(df, target=target_col, protected_attributes=protected, test_size=test_size)
                invalidate_after_data_load(st, target_col=target_col)
                st.toast("Өгөгдөл боловсруулагдлаа")
                st.rerun()
            except Exception as e:
                _handle_action_error(st, "Өгөгдөл боловсруулах", e)

    if st.session_state.get('data_loaded') and fw and fw.X_train is not None:
        st.markdown(f"""<div class="a-ok">
            Сургалт: {len(fw.X_train):,} · Тест: {len(fw.X_test):,} · Features: {len(fw.feature_names)}
        </div>""", unsafe_allow_html=True)
        processing_info = getattr(fw, 'processing_info', None) or st.session_state.get('processing_info')
        if processing_info:
            with st.expander("Processing recipe"):
                rows = [
                    {"Field": "Missing strategy", "Value": processing_info.get('missing_strategy')},
                    {"Field": "Encoding", "Value": processing_info.get('encoding_method')},
                    {"Field": "Normalization", "Value": processing_info.get('normalization_method')},
                    {"Field": "Processed features", "Value": processing_info.get('n_processed_features')},
                    {"Field": "Original features", "Value": processing_info.get('n_original_features')},
                ]
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ============================================================================
# 2. DATA QUALITY SECTION
# ============================================================================
def render_data_quality_section():
    import streamlit as st
    fw = _check_framework(st)

    st.markdown('<div class="sec-head">Чанарын Шинжилгээ</div>', unsafe_allow_html=True)
    _section_intro(st, "Data Diagnostics", "Missing, leakage, imbalance, suitability check.")

    if st.session_state.get('uploaded_data') is None:
        _need(st, "Эхлээд өгөгдөл ачаална уу.")
        return

    df = st.session_state['uploaded_data']
    target_col = st.session_state.get('target_col')

    if st.button(f"{ICONS['play']} Шинжилгээ", type="primary", width="stretch", key="btn_quality"):
        with st.spinner("Чанарыг шинжилж байна..."):
            try:
                from src.data_processing.data_quality import DataQualityAnalyzer
                report = DataQualityAnalyzer().analyze(df, target=target_col)
                st.session_state['quality_report'] = report
                st.rerun()
            except Exception as e:
                _handle_action_error(st, "Өгөгдлийн чанарын шинжилгээ", e)

    report = st.session_state.get('quality_report')
    if report is None:
        return
    qs = report.get('quality_score', 0)
    if isinstance(qs, dict):
        qs = qs.get('score', 0)
    qs = float(qs or 0)

    missing_pct = report.get('missing_values', {}).get('total_missing_percent', 0)
    dup = report.get('duplicates', {}).get('duplicate_rows', 0)
    leakage = report.get('leakage_risks', {}).get('risk_count', 0)
    target_analysis = report.get('target_analysis')
    overview = report.get('overview', {})

    c1, c2, c3, c4 = st.columns(4)
    clr = "#22c55e" if qs >= 80 else "#eab308" if qs >= 60 else "#ef4444"
    c1.markdown(f'<div class="m-card"><div class="m-val" style="color:{clr}">{qs:.0f}</div><div class="m-lbl">Чанарын Оноо</div></div>', unsafe_allow_html=True)
    c2.metric("Дутуу утга", f"{missing_pct:.1f}%")
    c3.metric("Давхардал", dup)
    c4.metric("Leakage risk", leakage)

    if overview.get('size_message'):
        st.markdown(f'<div class="a-warn">{html.escape(overview["size_message"])}</div>', unsafe_allow_html=True)

    if target_analysis is not None:
        target_issue_count = len(target_analysis.get('issues', []))
        status_class = 'a-ok' if target_analysis.get('is_suitable', False) else 'a-warn'
        st.markdown(
            f'<div class="{status_class}"><strong>Target profile</strong><br/>{html.escape(target_analysis.get("task_type", "unknown"))} · '
            f'{target_analysis.get("unique_count", 0)} unique · {target_issue_count} issue</div>',
            unsafe_allow_html=True,
        )

    if report.get('leakage_risks', {}).get('high_risk_count', 0) > 0:
        st.markdown('<div class="a-err"><strong>Leakage warning</strong><br/>Target-тай шууд холбоотой байж болох feature илэрсэн тул training-ээс өмнө заавал шалгана уу.</div>', unsafe_allow_html=True)

    # Details table
    missing_det = report.get('missing_values', {}).get('details', {})
    if missing_det:
        with st.expander("Дутуу утгатай баганууд"):
            rows = [{"Багана": c, "Тоо": info['count'], "%": f"{info['percent']:.1f}"} for c, info in missing_det.items()]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if target_analysis is not None:
        with st.expander("Target suitability"):
            info_rows = [
                {"Field": "Target", "Value": target_analysis.get('target')},
                {"Field": "dtype", "Value": target_analysis.get('dtype')},
                {"Field": "Task type", "Value": target_analysis.get('task_type')},
                {"Field": "Recommended metric", "Value": target_analysis.get('recommended_metric')},
                {"Field": "Unique values", "Value": target_analysis.get('unique_count')},
                {"Field": "Missing %", "Value": f"{target_analysis.get('missing_percent', 0):.1f}"},
            ]
            st.dataframe(pd.DataFrame(info_rows), width="stretch", hide_index=True)
            if target_analysis.get('issues'):
                for issue in target_analysis['issues']:
                    cls = 'a-err' if issue['severity'] == 'high' else 'a-warn' if issue['severity'] == 'medium' else 'a-info'
                    st.markdown(f'<div class="{cls}">{html.escape(issue["message"])}</div>', unsafe_allow_html=True)

    structural = report.get('structural_risks', {})
    structural_rows = []
    for item in structural.get('numeric_text_columns', []):
        structural_rows.append({"Type": "Numeric as text", "Column": item['column'], "Evidence": f"ratio={item['numeric_ratio']:.1%}"})
    for item in structural.get('mixed_type_columns', []):
        structural_rows.append({"Type": "Mixed type", "Column": item['column'], "Evidence": f"numeric ratio={item['numeric_ratio']:.1%}"})
    for item in structural.get('date_like_columns', []):
        structural_rows.append({"Type": "Date-like text", "Column": item['column'], "Evidence": f"parsed ratio={item['parsed_ratio']:.1%}"})
    for item in structural.get('identifier_like_columns', []):
        structural_rows.append({"Type": "Identifier-like", "Column": item['column'], "Evidence": f"unique ratio={item['unique_ratio']:.1%}"})
    if structural_rows:
        with st.expander("Structural warnings"):
            st.dataframe(pd.DataFrame(structural_rows), width="stretch", hide_index=True)

    leakage_rows = []
    for item in report.get('leakage_risks', {}).get('details', []):
        leakage_rows.append({
            "Feature": item['feature'],
            "Signal": item['risk_type'],
            "Severity": item['severity'],
            "Evidence": item['evidence'],
        })
    if leakage_rows:
        with st.expander("Leakage heuristic signals"):
            st.dataframe(pd.DataFrame(leakage_rows), width="stretch", hide_index=True)

    class_balance = report.get('class_balance')
    if class_balance:
        with st.expander("Class balance"):
            rows = [
                {"Class": str(label), "Count": count, "%": f"{class_balance['class_percentages'][label]:.1f}"}
                for label, count in class_balance['class_distribution'].items()
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    recs = report.get('recommendations', [])
    if recs:
        with st.expander("Зөвлөмж"):
            for r in recs:
                if isinstance(r, dict):
                    msg = r.get('message', '')
                    action = r.get('action', '')
                    priority = r.get('priority', 'low')
                    cls = 'a-err' if priority == 'high' else 'a-warn' if priority == 'medium' else 'a-info'
                    st.markdown(
                        f'<div class="{cls}"><strong>{html.escape(msg)}</strong><br/>{html.escape(action)}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"· {r}")


# ============================================================================
# 3. MODEL TRAINING SECTION
# ============================================================================
def render_model_section():
    import streamlit as st
    import time

    fw = _check_framework(st)
    st.markdown('<div class="sec-head">Загвар Сургах</div>', unsafe_allow_html=True)
    _section_intro(st, "Experiment Lab", "Train, compare, choose the active model.")

    if not st.session_state.get('data_loaded'):
        _need(st, "Эхлээд өгөгдөл ачааллаж боловсруулна уу.")
        return

    ALL_MODELS = {
        "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
        "random_forest": "Random Forest", "extra_trees": "Extra Trees",
        "gradient_boosting": "Gradient Boosting", "adaboost": "AdaBoost",
        "logistic_regression": "Logistic Regression", "svm": "SVM"
    }

    selected = st.multiselect("Загварууд сонгох", list(ALL_MODELS.keys()),
                               default=["xgboost"], format_func=lambda x: ALL_MODELS[x])

    validation_cols = st.columns(2)
    validation_strategy = validation_cols[0].selectbox(
        "Validation strategy",
        ["holdout", "holdout_cv"],
        format_func=lambda value: {
            'holdout': 'Holdout only',
            'holdout_cv': 'Holdout + k-fold CV',
        }[value],
    )
    cv_folds = validation_cols[1].slider(
        "CV folds",
        min_value=3,
        max_value=10,
        value=5,
        disabled=validation_strategy == 'holdout',
    )

    with st.expander(f"{ICONS['settings']} Hyperparameters"):
        c1, c2, c3 = st.columns(3)
        n_est = c1.slider("n_estimators", 50, 500, 100)
        m_dep = c2.slider("max_depth", 2, 15, 6)
        lr = c3.slider("learning_rate", 0.01, 0.3, 0.1)

    c1, c2 = st.columns(2)
    go = c1.button(f"{ICONS['play']} Сургах", type="primary", width="stretch", disabled=not selected)
    if c2.button(f"{ICONS['refresh']} Арилгах", width="stretch"):
        clear_model_state(st)
        st.rerun()

    if go and selected:
        bar = st.progress(0)
        trained = st.session_state.get('trained_models', {})
        for i, model_type in enumerate(selected):
            bar.progress(i / len(selected), f"{ALL_MODELS[model_type]} сургаж байна...")
            try:
                t0 = time.time()
                params = {}
                if model_type in ['xgboost', 'lightgbm', 'catboost', 'gradient_boosting']:
                    params = {'n_estimators': n_est, 'max_depth': m_dep, 'learning_rate': lr}
                elif model_type in ['random_forest', 'extra_trees']:
                    params = {'n_estimators': n_est, 'max_depth': m_dep}
                elif model_type == 'adaboost':
                    params = {'n_estimators': n_est, 'learning_rate': lr}

                fw.train_model(model_type=model_type, **params)
                dt = time.time() - t0
                train_metrics, is_cls = _compute_model_metrics(fw.model, fw.X_train, fw.y_train)
                test_metrics, _ = _compute_model_metrics(fw.model, fw.X_test, fw.y_test)
                overfit_gap = train_metrics['primary_metric'] - test_metrics['primary_metric']
                cv_summary = None
                selection_score = float(test_metrics['primary_metric'])
                selection_basis = f"Holdout {test_metrics['primary_metric_name'].upper()}"

                if validation_strategy == 'holdout_cv':
                    cv_summary = _compute_cross_validation_summary(
                        fw.config,
                        model_type,
                        params,
                        fw.X_train,
                        fw.y_train,
                        cv_folds,
                    )
                    if cv_summary is not None:
                        selection_score = float(cv_summary['mean'])
                        selection_basis = f"{cv_summary['folds']}-fold CV {cv_summary['metric_name'].upper()}"

                trained[model_type] = {
                    'model': fw.model,
                    'metrics': test_metrics,
                    'train_metrics': train_metrics,
                    'cv_summary': cv_summary,
                    'overfit_gap': overfit_gap,
                    'selection_score': selection_score,
                    'selection_basis': selection_basis,
                    'train_time': dt,
                    'is_classification': is_cls, 'params': params,
                }
            except Exception as e:
                _handle_action_error(st, f"{ALL_MODELS[model_type]} загвар сургах", e)

        bar.progress(1.0, "Дууслаа")
        st.session_state['trained_models'] = trained
        if trained:
            best = max(trained, key=lambda x: trained[x].get('selection_score', float('-inf')))
            fw.train_model(model_type=best, **trained[best]['params'])
            st.session_state['selected_model_name'] = best
            invalidate_after_model_change(st)
        else:
            clear_model_state(st)
        st.rerun()

    # Results
    trained = st.session_state.get('trained_models', {})
    if not trained:
        return

    st.markdown("---")
    is_cls = list(trained.values())[0]['is_classification']

    if is_cls:
        import plotly.graph_objects as go
        rows = []
        for n, d in trained.items():
            m = d['metrics']
            train_m = d.get('train_metrics', {})
            cv_summary = d.get('cv_summary') or {}
            rows.append({
                'Загвар': n,
                'Train F1': train_m.get('f1', 0),
                'Test F1': m.get('f1', 0),
                'CV F1': cv_summary.get('mean'),
                'CV std': cv_summary.get('std'),
                'Accuracy': m.get('accuracy', 0),
                'Precision': m.get('precision', 0), 'Recall': m.get('recall', 0),
                'Gap': d.get('overfit_gap', 0),
                'AUC': m.get('auc', None), 'Хугацаа': f"{d['train_time']:.2f}s",
            })
        df_m = pd.DataFrame(rows)

        # Styled table
        st.dataframe(
            df_m.style.format({
                'Train F1': '{:.4f}', 'Test F1': '{:.4f}', 'Accuracy': '{:.4f}',
                'CV F1': lambda x: f'{x:.4f}' if pd.notnull(x) else '—',
                'CV std': lambda x: f'{x:.4f}' if pd.notnull(x) else '—',
                'Precision': '{:.4f}', 'Recall': '{:.4f}', 'Gap': '{:.4f}',
                'AUC': lambda x: f'{x:.4f}' if x else '—',
            }).highlight_max(subset=['Test F1', 'Accuracy', 'Precision', 'Recall'], color='rgba(59,130,246,0.15)'),
            width="stretch", hide_index=True,
        )

        # Chart
        names = list(trained.keys())
        fig = go.Figure()
        for metric, color in [('accuracy', '#3b82f6'), ('f1', '#22c55e'), ('precision', '#eab308'), ('recall', '#8b5cf6')]:
            vals = [trained[n]['metrics'].get(metric, 0) for n in names]
            fig.add_trace(go.Bar(name=metric.upper(), x=names, y=vals, marker_color=color,
                                 text=[f'{v:.3f}' for v in vals], textposition='auto'))
        fig.update_layout(barmode='group', height=350, yaxis_range=[0, 1], title_text="Загваруудын Харьцуулалт")
        _plotly_chart(st, fig)

        gap_rows = []
        for name, details in trained.items():
            gap = details.get('overfit_gap', 0)
            risk = 'high' if gap >= 0.10 else 'medium' if gap >= 0.05 else 'low'
            cv_summary = details.get('cv_summary') or {}
            gap_rows.append({
                'Загвар': name,
                'Train-Test gap': gap,
                'Overfitting risk': risk,
                'Selection basis': details.get('selection_basis', 'Holdout'),
                'CV folds': cv_summary.get('folds', '—'),
            })
        st.dataframe(pd.DataFrame(gap_rows), width="stretch", hide_index=True)

        best = df_m.loc[df_m['Test F1'].idxmax(), 'Загвар']
        best = max(trained, key=lambda name: trained[name].get('selection_score', float('-inf')))
        best_gap = trained[best].get('overfit_gap', 0)
        gap_class = 'a-warn' if best_gap >= 0.05 else 'a-ok'
        st.markdown(
            f'<div class="{gap_class}">Шилдэг загвар: <strong>{best}</strong> · {trained[best].get("selection_basis", "Holdout")} · Test F1={trained[best]["metrics"].get("f1", 0):.4f} · gap={best_gap:.4f}</div>',
            unsafe_allow_html=True,
        )
    else:
        rows = [{'Загвар': n, 'Train R²': d.get('train_metrics', {}).get('r2', 0), 'Test R²': d['metrics'].get('r2', 0),
                 'CV R²': (d.get('cv_summary') or {}).get('mean'), 'CV std': (d.get('cv_summary') or {}).get('std'),
                 'Gap': d.get('overfit_gap', 0), 'RMSE': d['metrics'].get('rmse', 0),
                 'MAE': d['metrics'].get('mae', 0), 'Хугацаа': f"{d['train_time']:.2f}s"}
                for n, d in trained.items()]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # Model selector for SHAP
    st.markdown("---")
    active = st.selectbox("SHAP-д ашиглах загвар", list(trained.keys()))
    if st.button(f"{ICONS['check']} Сонгох", type="primary"):
        fw.train_model(model_type=active, **trained[active]['params'])
        st.session_state['selected_model_name'] = active
        invalidate_after_model_change(st)
        st.rerun()

    sel = st.session_state.get('selected_model_name')
    if sel:
        st.markdown(f'<div class="a-ok"><strong>{sel}</strong> сонгогдсон</div>', unsafe_allow_html=True)


# ============================================================================
# 4. MODEL METRICS SECTION
# ============================================================================
def render_model_metrics_section():
    import streamlit as st
    fw = _check_framework(st)

    st.markdown('<div class="sec-head">Загварын Гүйцэтгэл</div>', unsafe_allow_html=True)
    _section_intro(st, "Performance Review", "Threshold, confusion, calibration, subgroup performance.")

    if not st.session_state.get('model_trained'):
        _need(st, "Эхлээд загвар сургаарай.")
        return

    active_threshold = float(st.session_state.get('decision_threshold', 0.5))
    probability_scores = get_binary_probability_scores(fw.model, fw.X_test)
    supports_threshold_control = probability_scores is not None and len(np.unique(fw.y_test)) == 2

    if supports_threshold_control:
        threshold_cols = st.columns([2, 1])
        threshold_cols[0].slider(
            "Active decision threshold",
            min_value=0.05,
            max_value=0.95,
            value=active_threshold,
            step=0.01,
            key='decision_threshold',
        )
        threshold_cols[1].metric("Current threshold", f"{st.session_state['decision_threshold']:.2f}")
        active_threshold = float(st.session_state.get('decision_threshold', 0.5))

    if st.button(f"{ICONS['play']} Графикууд", type="primary", width="stretch", key="btn_metrics"):
        with st.spinner("Графикуудыг үүсгэж байна..."):
            try:
                from src.visualization.metrics_viz import MetricsVisualizer
                vis = MetricsVisualizer()
                yt = fw.y_test
                ypr = probability_scores
                yp = predict_with_threshold(fw.model, fw.X_test, active_threshold if supports_threshold_control else None)

                threshold_summary = _find_best_classification_threshold(yt, ypr) if ypr is not None else None

                tabs = st.tabs(["ROC", "PR", "Confusion", "Threshold", "Calibration", "Distribution"])
                charts = [
                    (0, lambda: vis.plot_roc_curve(yt, ypr) if ypr is not None else None),
                    (1, lambda: vis.plot_pr_curve(yt, ypr) if ypr is not None else None),
                    (2, lambda: vis.plot_confusion_matrix(yt, yp, ['Negative', 'Positive'], normalize=True)),
                    (3, lambda: vis.plot_threshold_analysis(yt, ypr) if ypr is not None else None),
                    (4, lambda: vis.plot_calibration_curve(yt, ypr) if ypr is not None else None),
                    (5, lambda: vis.plot_prediction_distribution(yt, ypr) if ypr is not None else None),
                ]
                for idx, fn in charts:
                    with tabs[idx]:
                        fig = fn()
                        if fig:
                            _plotly_chart(st, fig)
                            if idx == 3 and threshold_summary is not None:
                                st.markdown(
                                    f'<div class="a-ok"><strong>Suggested threshold</strong><br/>'
                                    f't={threshold_summary["threshold"]:.2f} · '
                                    f'F1={threshold_summary["f1"]:.3f} · '
                                    f'Precision={threshold_summary["precision"]:.3f} · '
                                    f'Recall={threshold_summary["recall"]:.3f} · '
                                    f'Active={active_threshold:.2f}</div>',
                                    unsafe_allow_html=True,
                                )
                                if st.button("Suggested threshold ашиглах", key="apply_suggested_threshold", width="stretch"):
                                    st.session_state['decision_threshold'] = float(threshold_summary['threshold'])
                                    st.rerun()
                        else:
                            st.info("Probability шаардлагатай")
            except Exception as e:
                _handle_action_error(st, "Гүйцэтгэлийн график үүсгэх", e)

    protected_test_data = getattr(fw, '_protected_test_data', None)
    protected_attributes = getattr(fw, '_protected_attributes', []) or []
    available_comparison_attrs = [attr for attr in protected_attributes if attr in getattr(protected_test_data, 'columns', [])]
    if (
        protected_test_data is not None
        and available_comparison_attrs
        and len(protected_test_data) == len(fw.X_test)
    ):
        st.markdown("#### Group-wise Metric Comparison")
        comparison_attr = st.selectbox(
            "Protected attribute for performance review",
            available_comparison_attrs,
            key='performance_group_attribute',
        )

        if comparison_attr:
            try:
                from src.evaluation.fairness import FairnessEvaluator
                comparison_result = FairnessEvaluator().evaluate(
                    model=fw.model,
                    X_test=fw.X_test,
                    y_test=fw.y_test,
                    protected_attributes=[comparison_attr],
                    protected_data=protected_test_data[[comparison_attr]],
                    decision_threshold=active_threshold if supports_threshold_control else None,
                )
                attr_metrics = comparison_result.get('metrics_by_attribute', {}).get(comparison_attr, {})
                group_rows = _build_group_metric_rows(attr_metrics.get('group_metrics', {}))

                if group_rows:
                    summary_class = 'a-ok' if attr_metrics.get('is_fair') else 'a-warn'
                    st.markdown(
                        f'<div class="{summary_class}"><strong>{html.escape(comparison_attr)}</strong><br/>'
                        f'Accuracy gap={attr_metrics.get("accuracy_gap", 0):.3f} · '
                        f'F1 gap={attr_metrics.get("f1_gap", 0):.3f} · '
                        f'DP ratio={attr_metrics.get("demographic_parity_ratio", 0):.3f}</div>',
                        unsafe_allow_html=True,
                    )

                    group_df, display_df = _format_group_metric_display(group_rows)
                    st.dataframe(display_df, width="stretch", hide_index=True)

                    try:
                        import plotly.graph_objects as go

                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=group_df['Бүлэг'],
                            y=group_df['Accuracy'],
                            name='Accuracy',
                            marker_color='#3b82f6',
                            text=[f'{value:.2%}' for value in group_df['Accuracy']],
                            textposition='auto',
                        ))
                        fig.add_trace(go.Bar(
                            x=group_df['Бүлэг'],
                            y=group_df['F1'],
                            name='F1',
                            marker_color='#22c55e',
                            text=[f'{value:.2%}' for value in group_df['F1']],
                            textposition='auto',
                        ))
                        fig.update_layout(
                            title_text=f'{comparison_attr} by group',
                            barmode='group',
                            yaxis_range=[0, 1],
                            height=320,
                        )
                        _plotly_chart(st, fig)
                    except Exception as e:
                        _handle_action_error(st, 'Group-wise metric chart', e)
            except Exception as e:
                _handle_action_error(st, 'Group-wise metric comparison', e)


# ============================================================================
# 5. SHAP EXPLANATIONS SECTION
# ============================================================================
def render_explanation_section():
    import streamlit as st
    fw = _check_framework(st)

    st.markdown('<div class="sec-head">SHAP Тайлбарууд</div>', unsafe_allow_html=True)
    _section_intro(st, "Explanation Core", "Inspect global and local SHAP drivers.")

    if not st.session_state.get('model_trained'):
        _need(st, "Эхлээд загвар сургаарай.")
        return

    # Sync SHAP from session
    expl = st.session_state.get('explanations', {})
    if fw and fw.shap_values is None and 'shap_values' in expl:
        fw.shap_values = expl['shap_values']
        fw._explanations = expl

    c1, c2 = st.columns(2)
    exp_type = c1.selectbox("Төрөл", ["both", "global", "local"],
                             format_func=lambda x: {"both": "Global + Local", "global": "Global", "local": "Local"}[x])
    max_f = c2.slider("Features тоо", 5, 30, 20)

    if st.button(f"{ICONS['play']} SHAP Үүсгэх", type="primary", width="stretch", key="btn_shap"):
        with st.spinner("SHAP утгуудыг тооцоолж байна..."):
            try:
                expl = fw.explain(explanation_type=exp_type)
                st.session_state['explanations'] = expl
                st.session_state['explanations_generated'] = True
                st.toast("SHAP тайлбарууд үүслээ")
                st.rerun()
            except Exception as e:
                _handle_action_error(st, "SHAP тайлбар үүсгэх", e)

    expl = st.session_state.get('explanations', {})
    has_shap = (fw and fw.shap_values is not None) or ('shap_values' in expl)
    if not has_shap:
        return

    shap_values_array = _get_active_shap_values(fw, expl)
    feature_names = expl.get('feature_names') or fw.feature_names or []
    protected_attributes = getattr(fw, '_protected_attributes', []) or []

    with st.expander("Explanation metadata"):
        explainer_object = expl.get('explainer')
        metadata_rows = [
            {'Field': 'Active model', 'Value': st.session_state.get('selected_model_name') or type(fw.model).__name__},
            {'Field': 'Explainer', 'Value': type(explainer_object).__name__ if explainer_object is not None else 'Unknown'},
            {'Field': 'Explained rows', 'Value': int(shap_values_array.shape[0]) if shap_values_array is not None and shap_values_array.ndim >= 1 else 0},
            {'Field': 'Explained features', 'Value': int(shap_values_array.shape[1]) if shap_values_array is not None and shap_values_array.ndim >= 2 else len(feature_names)},
            {'Field': 'Base value reference', 'Value': _resolve_base_value(expl.get('base_value'))},
            {'Field': 'Global block', 'Value': 'available' if 'global' in expl else 'missing'},
            {'Field': 'Local block', 'Value': 'available' if 'local' in expl else 'missing'},
            {'Field': 'Protected attributes', 'Value': ', '.join(map(str, protected_attributes)) if protected_attributes else 'none'},
        ]
        st.dataframe(pd.DataFrame(metadata_rows), width="stretch", hide_index=True)
        global_summary = expl.get('global', {}).get('summary')
        if global_summary:
            st.markdown(
                f'<div class="intro-strip"><span class="intro-label">Global narrative</span><span class="intro-text">{html.escape(global_summary)}</span></div>',
                unsafe_allow_html=True,
            )

    # Global
    if 'global' in expl:
        st.markdown("#### Global Feature Importance")
        imp_data = expl['global'].get('feature_importance', [])
        if isinstance(imp_data, list) and imp_data:
            df_imp = pd.DataFrame(imp_data)
            st.dataframe(df_imp.head(max_f), width="stretch", hide_index=True)

    # Local
    if 'local' in expl:
        st.markdown("#### Local Тайлбар")
        local_list = expl['local'].get('explanations', [])
        if local_list:
            active_threshold = float(st.session_state.get('decision_threshold', 0.5))
            browser_index = st.slider("Дээж", 0, len(local_list) - 1, 0)
            e = local_list[browser_index]
            sample_index = min(int(e.get('sample_index', browser_index)), len(fw.X_test) - 1)
            sample_context = _get_sample_prediction_context(fw, sample_index, active_threshold)

            contribution_rows = []
            for item in e.get('contributions', [])[:max_f]:
                shap_value = float(item.get('shap_value', 0))
                contribution_rows.append({
                    'Feature': item.get('feature'),
                    'Value': item.get('feature_value'),
                    'SHAP': shap_value,
                    'Impact': item.get('impact'),
                    'Magnitude': abs(shap_value),
                })

            summary_cols = st.columns(4)
            summary_cols[0].metric('Sample', f'#{sample_index}')
            summary_cols[1].metric('Base value', f"{e.get('base_value', 0):.4f}")
            summary_cols[2].metric('Prediction', _fmt_val(sample_context['predicted_value']))
            summary_cols[3].metric('Actual', _fmt_val(sample_context['actual_value']) if sample_context['actual_value'] is not None else '—')

            if sample_context['predicted_probability'] is not None:
                st.markdown(
                    f'<div class="a-info"><strong>Predicted probability</strong><br/>{sample_context["predicted_probability"]:.4f} · Net SHAP shift = {sum(row["SHAP"] for row in contribution_rows):+.4f}</div>',
                    unsafe_allow_html=True,
                )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Эерэг нөлөө**")
                pos_data = []
                for it in e.get('top_positive', [])[:5]:
                    sv = it.get('shap_value', it.get('contribution', 0))
                    pos_data.append({"Feature": it['feature'], "SHAP": f"+{sv:.4f}"})
                if pos_data:
                    st.dataframe(pd.DataFrame(pos_data), width="stretch", hide_index=True)
            with c2:
                st.markdown("**Сөрөг нөлөө**")
                neg_data = []
                for it in e.get('top_negative', [])[:5]:
                    sv = it.get('shap_value', it.get('contribution', 0))
                    neg_data.append({"Feature": it['feature'], "SHAP": f"{sv:.4f}"})
                if neg_data:
                    st.dataframe(pd.DataFrame(neg_data), width="stretch", hide_index=True)

            if contribution_rows:
                st.markdown("**Top contribution details**")
                st.dataframe(
                    pd.DataFrame(contribution_rows),
                    width="stretch",
                    hide_index=True,
                )

            if shap_values_array is not None and len(local_list) > 1:
                with st.expander("Local compare mode"):
                    compare_cols = st.columns(2)
                    compare_left = compare_cols[0].selectbox(
                        'Sample A',
                        list(range(len(local_list))),
                        format_func=lambda index: f"#{local_list[index].get('sample_index', index)}",
                        key='local_compare_a',
                    )
                    right_options = [index for index in range(len(local_list)) if index != compare_left]
                    compare_right = compare_cols[1].selectbox(
                        'Sample B',
                        right_options,
                        format_func=lambda index: f"#{local_list[index].get('sample_index', index)}",
                        key='local_compare_b',
                    )

                    sample_a_index = min(int(local_list[compare_left].get('sample_index', compare_left)), len(shap_values_array) - 1)
                    sample_b_index = min(int(local_list[compare_right].get('sample_index', compare_right)), len(shap_values_array) - 1)
                    sample_a_context = _get_sample_prediction_context(fw, sample_a_index, active_threshold)
                    sample_b_context = _get_sample_prediction_context(fw, sample_b_index, active_threshold)

                    compare_summary_cols = st.columns(2)
                    compare_summary_cols[0].markdown(
                        f'<div class="a-info"><strong>Sample A</strong><br/>Prediction: {_fmt_val(sample_a_context["predicted_value"])} · Actual: {_fmt_val(sample_a_context["actual_value"])} </div>',
                        unsafe_allow_html=True,
                    )
                    compare_summary_cols[1].markdown(
                        f'<div class="a-info"><strong>Sample B</strong><br/>Prediction: {_fmt_val(sample_b_context["predicted_value"])} · Actual: {_fmt_val(sample_b_context["actual_value"])} </div>',
                        unsafe_allow_html=True,
                    )

                    compare_rows = _build_local_compare_rows(
                        shap_values_array,
                        feature_names,
                        sample_a_index,
                        sample_b_index,
                        max_f,
                    )
                    st.dataframe(pd.DataFrame(compare_rows), width="stretch", hide_index=True)


    protected_test_data = getattr(fw, '_protected_test_data', None)
    if (
        shap_values_array is not None
        and protected_test_data is not None
        and len(protected_test_data) == len(shap_values_array)
        and len(protected_test_data.columns) > 0
    ):
        st.markdown("#### Subgroup Explanation Summary")
        subgroup_attribute = st.selectbox(
            'Protected/group attribute',
            protected_test_data.columns.tolist(),
            key='shap_subgroup_attribute',
        )
        subgroup_rows = _build_subgroup_shap_rows(
            fw,
            shap_values_array,
            feature_names,
            subgroup_attribute,
        )
        if subgroup_rows:
            st.dataframe(pd.DataFrame(subgroup_rows), width="stretch", hide_index=True)


    # Math (collapsed)
    with st.expander("SHAP Математик"):
        st.latex(r'\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [f(S \cup \{i\}) - f(S)]')
        st.caption("phi_i > 0: өсгөнө | phi_i < 0: бууруулна")


# ============================================================================
# 6. VISUALIZATION SECTION
# ============================================================================
def render_visualization_section():
    import streamlit as st
    fw = _check_framework(st)

    st.markdown('<div class="sec-head">Визуализаци</div>', unsafe_allow_html=True)
    _section_intro(st, "Figure Studio", "Generate export-ready SHAP figures.")

    expl = st.session_state.get('explanations', {})
    if fw and fw.shap_values is None and 'shap_values' in expl:
        fw.shap_values = expl['shap_values']
        fw._explanations = expl

    has_shap = (fw and fw.shap_values is not None) or ('shap_values' in expl)
    if not st.session_state.get('explanations_generated') or not has_shap:
        _need(st, "Эхлээд SHAP тайлбар үүсгэнэ үү.")
        return

    PLOTS = {
        "summary": "Beeswarm", "bar": "Bar", "waterfall": "Waterfall",
        "heatmap": "Heatmap", "violin": "Violin", "dependence": "Dependence",
    }

    c1, c2 = st.columns([1, 2])
    pt = c1.selectbox("График", list(PLOTS.keys()), format_func=lambda x: PLOTS[x])
    mx = c2.slider("Features", 5, 30, 15)

    kw = {'max_display': mx}
    sidx = 0
    if pt == "waterfall":
        sidx = st.slider("Дээж", 0, len(fw.X_test) - 1, 0, key="wf_idx")
        kw['sample_idx'] = sidx
        sv = fw.shap_values if fw.shap_values is not None else expl.get('shap_values')
        if sv is not None:
            kw['base_value'] = float(np.mean(sv.sum(axis=1)))
    if pt == "dependence":
        feat = st.selectbox("Feature", fw.feature_names, key="dep_feat")
        kw['feature'] = feat

    if st.button(f"{ICONS['play']} Үүсгэх", type="primary", width="stretch", key="btn_viz"):
        with st.spinner(f"{PLOTS[pt]} үүсгэж байна..."):
            try:
                fig = fw.visualize(plot_type=pt, **kw)
                _plotly_chart(st, fig)
            except Exception as e:
                _handle_action_error(st, "SHAP визуализаци үүсгэх", e)


# ============================================================================
# 7. ADVANCED ANALYSIS SECTION
# ============================================================================
def render_advanced_analysis_section():
    import streamlit as st
    fw = _check_framework(st)

    st.markdown('<div class="sec-head">Гүнзгий Шинжилгээ</div>', unsafe_allow_html=True)
    _section_intro(st, "Research Diagnostics", "Insights, stability, counterfactual, error analysis.")

    if fw is None or fw.shap_values is None:
        _need(st, "Эхлээд SHAP тайлбар үүсгэнэ үү.")
        return

    analysis_specs = _get_advanced_analysis_specs()
    analysis_results = st.session_state.setdefault('advanced_analysis_results', {})
    active_threshold = float(st.session_state.get('decision_threshold', 0.5))

    c1, c2, c3 = st.columns(3)
    c1.metric("Diagnostic modules", len(analysis_specs))
    c2.metric("Completed", len([name for name, value in analysis_results.items() if value is not None]))
    if get_binary_probability_scores(fw.model, fw.X_test) is not None and len(np.unique(fw.y_test)) == 2:
        c3.metric("Decision threshold", f"{active_threshold:.2f}")
    else:
        c3.metric("Decision threshold", "Model default")

    analysis_items = list(analysis_specs.items())
    for start in range(0, len(analysis_items), 2):
        cols = st.columns(2)
        for col, (analysis_name, meta) in zip(cols, analysis_items[start:start + 2]):
            status_label, status_tone = _get_analysis_status_meta(analysis_name, analysis_results)
            with col:
                _render_focus_card(
                    st,
                    meta['kicker'],
                    analysis_name,
                    meta['question'],
                    tone=status_tone,
                    footer=f"{meta['output']} {meta['next_step']}",
                    badge=status_label,
                )

    at = st.radio(
        "Diagnostic lane",
        list(analysis_specs.keys()),
        horizontal=True,
        key='advanced_analysis_type',
    )
    selected_meta = analysis_specs[at]
    selected_status, selected_tone = _get_analysis_status_meta(at, analysis_results)
    _render_focus_card(
        st,
        selected_meta['kicker'],
        at,
        selected_meta['question'],
        tone=selected_tone,
        footer=f"Output: {selected_meta['output']} Next: {selected_meta['next_step']}",
        badge=selected_status,
    )

    if at == "Counterfactual":
        _run_counterfactual(st, fw)
        return

    button_label = "Шинжилгээ шинэчлэх" if at in analysis_results else "Шинжилгээ эхлүүлэх"
    if st.button(f"{ICONS['play']} {button_label}", type="primary", width="stretch", key="btn_analysis"):
        with st.spinner(f"{at}..."):
            try:
                if at == "Автомат Дүгнэлт":
                    analysis_results[at] = _run_insights(fw, decision_threshold=active_threshold)
                elif at == "Тогтвортой Байдал":
                    analysis_results[at] = _run_stability(fw)
                elif at == "Алдааны Шинжилгээ":
                    analysis_results[at] = _run_error_analysis(fw, decision_threshold=active_threshold)
                st.session_state['advanced_analysis_results'] = analysis_results
            except Exception as e:
                _handle_action_error(st, "Гүнзгий шинжилгээ ажиллуулах", e)

    report = analysis_results.get(at)
    if report is None:
        _need(st, "Сонгосон diagnostic-ийг эхлүүлсний дараа structured result энд гарна.")
        return

    if at == "Автомат Дүгнэлт":
        _render_insights_report(st, report)
    elif at == "Тогтвортой Байдал":
        _render_stability_report(st, report)
    elif at == "Алдааны Шинжилгээ":
        _render_error_analysis_report(st, report)


def _run_insights(fw, decision_threshold=None):
    from src.analysis.insight_generator import InsightGenerator
    predictions = None
    try:
        if fw.y_test is not None and len(np.unique(fw.y_test)) == 2:
            predictions = predict_with_threshold(fw.model, fw.X_test, decision_threshold)
        elif hasattr(fw.model, 'predict'):
            predictions = np.asarray(fw.model.predict(fw.X_test))
    except Exception:
        predictions = None

    return InsightGenerator().generate(
        shap_values=fw.shap_values,
        X=fw.X_test,
        feature_names=fw.feature_names,
        predictions=predictions,
        y_true=fw.y_test,
    )


def _render_insights_report(st_module, report):
    imp = report.get('feature_importance', {})
    ranked = imp.get('ranked_features', [])
    top_feature = imp.get('top_feature') or {}
    threshold_info = report.get('thresholds', {})
    interaction_info = report.get('interactions', {})
    risk_factors = report.get('risk_factors', [])
    protective_factors = report.get('protective_factors', [])
    feature_effects = report.get('feature_effects', {})
    findings = report.get('key_findings', [])

    c1, c2, c3, c4 = st_module.columns(4)
    c1.metric("Top driver share", f"{top_feature.get('relative_percent', 0):.1f}%")
    c2.metric("Threshold clues", threshold_info.get('count', 0))
    c3.metric("Interactions", len(interaction_info.get('top_interactions', [])))
    c4.metric("Risk factors", len(risk_factors))

    if top_feature:
        st_module.markdown(
            f'<div class="a-ok"><strong>{html.escape(str(top_feature.get("feature", "Top feature")))}</strong><br/>'
            f'{html.escape(imp.get("insight", ""))}</div>',
            unsafe_allow_html=True,
        )
    _render_bullet_callout(st_module, "Executive findings", findings[:5], tone='info')

    tab1, tab2, tab3, tab4 = st_module.tabs([
        "Summary",
        "Thresholds & Interactions",
        "Risk & Protection",
        "Feature Effects",
    ])

    with tab1:
        if ranked:
            import plotly.graph_objects as go

            top = ranked[:10]
            fig = go.Figure(go.Bar(
                y=[item['feature'] for item in reversed(top)],
                x=[item['importance'] for item in reversed(top)],
                orientation='h',
                marker_color='#3b82f6',
                text=[f"{item['importance']:.4f}" for item in reversed(top)],
                textposition='auto',
            ))
            fig.update_layout(height=360, title_text="Feature Importance", xaxis_title="Mean |SHAP|")
            _plotly_chart(st_module, fig)

        summary_lines = _clean_summary_lines(report.get('summary', ''), strip_tags=True)
        if summary_lines:
            _render_bullet_callout(st_module, "Narrative summary", summary_lines[:6], tone='info')

    with tab2:
        threshold_rows = []
        for feature_name, details in list(threshold_info.get('detected_thresholds', {}).items())[:10]:
            threshold_rows.append({
                'Feature': feature_name,
                'Boundary': f"{details.get('threshold_value', 0):.3f}",
                'Direction': details.get('direction', '—'),
                'Confidence': f"{details.get('confidence', 0):.0%}",
                'Interpretation': details.get('description', '—'),
            })
        if threshold_rows:
            st_module.dataframe(pd.DataFrame(threshold_rows), width="stretch", hide_index=True)
        else:
            st_module.info("Тод threshold clue илрээгүй байна.")

        interaction_rows = []
        for interaction in interaction_info.get('top_interactions', []):
            interaction_rows.append({
                'Feature A': interaction.get('feature_1', '—'),
                'Feature B': interaction.get('feature_2', '—'),
                'Strength': f"{interaction.get('interaction_strength', 0):.4f}",
                'Direction': interaction.get('direction', '—'),
            })
        if interaction_rows:
            st_module.dataframe(pd.DataFrame(interaction_rows), width="stretch", hide_index=True)
        if interaction_info.get('insight'):
            st_module.markdown(f'<div class="a-info">{html.escape(interaction_info["insight"])}</div>', unsafe_allow_html=True)

    with tab3:
        left, right = st_module.columns(2)
        with left:
            st_module.markdown("#### Risk Factors")
            if risk_factors:
                risk_rows = [{
                    'Feature': item.get('feature', '—'),
                    'Impact': f"{item.get('avg_positive_impact', 0):.4f}",
                    'Frequency': f"{item.get('frequency', 0):.1%}",
                    'Typical value': f"{item.get('typical_risky_value', 0):.3f}",
                    'Description': item.get('description', '—'),
                } for item in risk_factors]
                st_module.dataframe(pd.DataFrame(risk_rows), width="stretch", hide_index=True)
            else:
                st_module.info("Тод risk factor илрээгүй байна.")

        with right:
            st_module.markdown("#### Protective Factors")
            if protective_factors:
                protective_rows = [{
                    'Feature': item.get('feature', '—'),
                    'Impact': f"{item.get('avg_negative_impact', 0):.4f}",
                    'Frequency': f"{item.get('frequency', 0):.1%}",
                    'Typical value': f"{item.get('typical_protective_value', 0):.3f}",
                    'Description': item.get('description', '—'),
                } for item in protective_factors]
                st_module.dataframe(pd.DataFrame(protective_rows), width="stretch", hide_index=True)
            else:
                st_module.info("Protective factor хүчтэй илрээгүй байна.")

    with tab4:
        effect_rows = []
        for feature_name, details in feature_effects.items():
            effect_rows.append({
                'Feature': feature_name,
                'Relationship': details.get('relationship_type', '—'),
                'Direction': details.get('direction', '—'),
                'Correlation': float(details.get('correlation_with_value', 0)),
                'Mean SHAP': float(details.get('mean_shap', 0)),
                'Std SHAP': float(details.get('std_shap', 0)),
            })
        if effect_rows:
            effect_df = pd.DataFrame(effect_rows)
            effect_df = effect_df.reindex(effect_df['Correlation'].abs().sort_values(ascending=False).index)
            display_df = effect_df.copy()
            display_df['Correlation'] = display_df['Correlation'].map(lambda value: f"{value:.3f}")
            display_df['Mean SHAP'] = display_df['Mean SHAP'].map(lambda value: f"{value:.4f}")
            display_df['Std SHAP'] = display_df['Std SHAP'].map(lambda value: f"{value:.4f}")
            st_module.dataframe(display_df.head(12), width="stretch", hide_index=True)
        else:
            st_module.info("Feature effect summary гарсангүй.")


def _run_stability(fw):
    from src.analysis.stability import StabilityAnalyzer
    return StabilityAnalyzer().analyze(
        model=fw.model, X=fw.X_test, feature_names=fw.feature_names,
        shap_values=fw.shap_values, n_iterations=30
    )


def _render_stability_report(st_module, report):
    if report.get('success') is False:
        st_module.warning(report.get('error', 'Алдаа'))
        return

    overall = report.get('overall_stability_score', report.get('overall_stability', 0))
    if isinstance(overall, dict):
        overall = overall.get('stability_score', 0)
    overall = float(overall)
    clr = "#22c55e" if overall >= 0.8 else "#eab308" if overall >= 0.6 else "#ef4444"

    feature_stability = report.get('feature_stability', {})
    stable_count = sum(int(details.get('is_stable', float(details.get('stability_score', 0)) >= 0.7)) for details in feature_stability.values())
    unstable_count = max(len(feature_stability) - stable_count, 0)
    ranking_info = report.get('rankings_stability') or {}

    c1, c2, c3, c4 = st_module.columns(4)
    c1.markdown(f'<div class="m-card"><div class="m-val" style="color:{clr}">{overall:.1%}</div><div class="m-lbl">Тогтвортой байдал</div></div>', unsafe_allow_html=True)
    c2.metric("Stable features", stable_count)
    c3.metric("Unstable features", unstable_count)
    c4.metric("Ranking consistency", f"{ranking_info.get('top_k_consistency', 0):.2f}")

    summary_lines = _clean_summary_lines(report.get('summary', ''), strip_tags=True)
    if summary_lines:
        tone = 'ok' if overall >= 0.8 else 'info' if overall >= 0.6 else 'warn'
        _render_bullet_callout(st_module, "Reliability summary", summary_lines[:5], tone=tone)

    fs = feature_stability
    if fs:
        import plotly.graph_objects as go

        rows = []
        confidence_intervals = report.get('confidence_intervals', {})
        for feature_name, details in fs.items():
            stability_score = float(details.get('stability_score', 0)) if isinstance(details, dict) else float(details)
            mean_importance = float(details.get('mean_importance', 0)) if isinstance(details, dict) else 0.0
            if feature_name in confidence_intervals:
                uncertainty = float(confidence_intervals[feature_name].get('ci_width', 0))
            else:
                uncertainty = float(details.get('std', details.get('std_across_folds', 0))) if isinstance(details, dict) else 0.0
            is_stable = details.get('is_stable', stability_score >= 0.7) if isinstance(details, dict) else stability_score >= 0.7
            rows.append({
                'Feature': feature_name,
                'Stability': stability_score,
                'Mean importance': mean_importance,
                'Uncertainty': uncertainty,
                'Status': 'Stable' if is_stable else 'Unstable',
            })

        stability_df = pd.DataFrame(rows).sort_values('Stability', ascending=True)
        fig = go.Figure(go.Bar(
            y=stability_df['Feature'].head(12),
            x=stability_df['Stability'].head(12),
            orientation='h',
            marker_color=['#22c55e' if value >= 0.7 else '#ef4444' for value in stability_df['Stability'].head(12)],
            text=[f"{value:.2f}" for value in stability_df['Stability'].head(12)],
            textposition='auto',
        ))
        fig.update_layout(height=380, title_text="Least stable features first", xaxis_range=[0, 1])
        _plotly_chart(st_module, fig)

        display_df = stability_df.copy()
        display_df['Stability'] = display_df['Stability'].map(lambda value: f"{value:.3f}")
        display_df['Mean importance'] = display_df['Mean importance'].map(lambda value: f"{value:.4f}")
        display_df['Uncertainty'] = display_df['Uncertainty'].map(lambda value: f"{value:.4f}")
        st_module.dataframe(display_df.head(12), width="stretch", hide_index=True)


def _run_counterfactual(st_module, fw):
    from src.analysis.counterfactual import CounterfactualGenerator

    st_module.markdown('<div class="intro-strip"><span class="intro-label">Counterfactual</span><span class="intro-text">Prediction өөрчлөгдөхөд хэрэгтэй хамгийн бага feature shift.</span></div>', unsafe_allow_html=True)

    gen = CounterfactualGenerator(model=fw.model, feature_names=fw.feature_names)

    c1, c2, c3 = st_module.columns(3)
    sidx = c1.number_input("Sample", 0, len(fw.X_test) - 1, 0)
    try:
        pred = fw.model.predict(fw.X_test[sidx].reshape(1, -1))[0]
        c2.metric("Одоогийн", f"Анги {int(round(pred))}")
    except Exception:
        c2.metric("Одоогийн", "—")
    tc = c3.selectbox("Зорилтот анги", [0, 1], index=1)

    if st_module.button(f"{ICONS['play']} Counterfactual", type="primary", width="stretch", key="btn_cf"):
        with st_module.spinner("Тооцоолж байна..."):
            try:
                cf = gen.generate(instance=fw.X_test[sidx], target_class=tc, max_changes=5, method='optimization')
                stored_result = {
                    'sample_index': int(sidx),
                    'target_class': int(tc),
                    'original_prediction': float(getattr(cf, 'original_prediction', 0)),
                    'counterfactual_prediction': float(getattr(cf, 'counterfactual_prediction', 0)),
                    'distance': float(getattr(cf, 'distance', 0)),
                    'validity': bool(getattr(cf, 'validity', False)),
                    'changes': _normalize_report_value(getattr(cf, 'changes', {})),
                }
                st_module.session_state.setdefault('advanced_analysis_results', {})['Counterfactual'] = stored_result
                if cf.changes:
                    rows = []
                    for feat, ch in cf.changes.items():
                        d = "▲" if ch['change'] > 0 else "▼"
                        rows.append({"Feature": feat, "Одоо": f"{ch['original']:.3f}",
                                     "Шинэ": f"{ch['new']:.3f}", "Чиглэл": d})
                    st_module.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                else:
                    st_module.info("Counterfactual олдсонгүй. Өөр sample сонгоно уу.")
            except Exception as e:
                _handle_action_error(st_module, "Counterfactual тооцоолох", e)


def _run_error_analysis(fw, decision_threshold=None):
    from src.analysis.error_analyzer import ErrorAnalyzer
    if fw.y_test is None or len(np.unique(fw.y_test)) != 2:
        raise ValueError("Алдааны шинжилгээ одоогоор binary classification workflow дээр дэмжигдэнэ.")

    return ErrorAnalyzer(
        model=fw.model,
        feature_names=fw.feature_names,
        threshold=decision_threshold or 0.5,
    ).analyze(
        X=fw.X_test, y_true=fw.y_test, shap_values=fw.shap_values
    )


def _render_error_analysis_report(st_module, report):
    high_severity_count = sum(int(pattern.severity == 'high') for pattern in report.patterns)
    c1, c2, c3, c4 = st_module.columns(4)
    c1.metric("Нийт алдаа", report.total_errors)
    c2.metric("FP rate", f"{report.fp_rate:.1%}")
    c3.metric("FN rate", f"{report.fn_rate:.1%}")
    c4.metric("High severity", high_severity_count)

    if report.total_errors == 0:
        st_module.markdown('<div class="a-ok"><strong>Error analysis</strong><br/>Алдаа илрээгүй байна. Threshold ба fairness review-г үргэлжлүүлэн шалгахад хангалттай.</div>', unsafe_allow_html=True)
    else:
        dominant_error = 'False Negative' if report.false_negatives > report.false_positives else 'False Positive'
        st_module.markdown(
            f'<div class="a-warn"><strong>Dominant error mode</strong><br/>{dominant_error} давамгай байна. Total errors={report.total_errors}.</div>',
            unsafe_allow_html=True,
        )

    tab1, tab2, tab3 = st_module.tabs(["Patterns", "Feature Drivers", "Cases & Actions"])

    with tab1:
        if report.patterns:
            rows = [{
                'Pattern': pattern.description,
                'Severity': pattern.severity,
                'Affected': pattern.affected_count,
                'Type': pattern.pattern_type,
            } for pattern in report.patterns[:8]]
            st_module.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st_module.info("Тод алдааны pattern илрээгүй байна.")

    with tab2:
        if report.feature_correlations:
            import plotly.graph_objects as go

            correlation_rows = sorted(
                report.feature_correlations.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:10]
            fig = go.Figure(go.Bar(
                y=[item[0] for item in reversed(correlation_rows)],
                x=[item[1] for item in reversed(correlation_rows)],
                orientation='h',
                marker_color=['#ef4444' if item[1] > 0 else '#3b82f6' for item in reversed(correlation_rows)],
                text=[f"{item[1]:.3f}" for item in reversed(correlation_rows)],
                textposition='auto',
            ))
            fig.update_layout(height=360, title_text="Features most correlated with errors")
            _plotly_chart(st_module, fig)

            corr_df = pd.DataFrame([
                {'Feature': feature, 'Correlation': f"{corr:.3f}"}
                for feature, corr in correlation_rows
            ])
            st_module.dataframe(corr_df, width="stretch", hide_index=True)
        else:
            st_module.info("Feature-error correlation summary алга байна.")

    with tab3:
        if report.error_cases:
            case_rows = [
                {
                    'Index': case.index,
                    'Type': case.error_type,
                    'True': case.true_label,
                    'Predicted': case.predicted_label,
                    'Confidence': f"{case.confidence:.3f}",
                }
                for case in report.error_cases[:12]
            ]
            st_module.dataframe(pd.DataFrame(case_rows), width="stretch", hide_index=True)
        _render_bullet_callout(st_module, "Recommended actions", report.recommendations[:6], tone='info')


# ============================================================================
# 8. FAIRNESS SECTION
# ============================================================================
def render_fairness_section():
    import streamlit as st
    fw = _check_framework(st)

    st.markdown('<div class="sec-head">Шударга Байдал</div>', unsafe_allow_html=True)
    _section_intro(st, "Responsible AI Check", "Audit subgroup gaps and fairness risk.")

    if not st.session_state.get('model_trained'):
        _need(st, "Эхлээд загвар сургаарай.")
        return

    active_threshold = float(st.session_state.get('decision_threshold', 0.5))
    if get_binary_probability_scores(fw.model, fw.X_test) is not None and len(np.unique(fw.y_test)) == 2:
        st.markdown(
            f'<div class="a-info"><strong>Decision threshold</strong><br/>Fairness analysis is using the active classification threshold: {active_threshold:.2f}</div>',
            unsafe_allow_html=True,
        )

    if st.button(f"{ICONS['play']} Шинжилгээ", type="primary", width="stretch", key="btn_fairness"):
        with st.spinner("Шинжилж байна..."):
            try:
                res = fw.evaluate(include_fairness=True, decision_threshold=active_threshold)
                if 'fairness' in res:
                    st.session_state['fairness_results'] = res['fairness']
                    st.rerun()
                else:
                    st.info("Хамгаалагдсан атрибут тохируулаагүй.")
            except Exception as e:
                _handle_action_error(st, "Шударга байдлын шинжилгээ", e)

    fr = st.session_state.get('fairness_results')
    if fr is None:
        return

    if 'warning' in fr:
        st.markdown(f'<div class="a-warn">{fr["warning"]}</div>', unsafe_allow_html=True)
        return

    if 'metrics_by_attribute' not in fr:
        return

    metrics_by_attribute = fr['metrics_by_attribute']
    is_fair = fr.get('overall_fairness', False)
    if is_fair:
        st.markdown('<div class="a-ok">Загвар шударга — бүх хэмжигдэхүүн хангагдсан.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="a-warn">Шударга бус байдал илэрлээ.</div>', unsafe_allow_html=True)

    audit_rows = _build_fairness_audit_rows(metrics_by_attribute)
    if not audit_rows:
        st.info("Protected attribute-аар fairness audit гарсангүй.")
        return

    flagged_count = sum(int(row['Status'] == 'Review') for row in audit_rows)
    largest_f1_gap = max(float(row['F1 Gap']) for row in audit_rows)
    top_priority = max(audit_rows, key=lambda row: row['F1 Gap'])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall status", "Pass" if is_fair else "Review")
    c2.metric("Attributes audited", len(audit_rows))
    c3.metric("Flagged", flagged_count)
    c4.metric("Largest F1 gap", f"{largest_f1_gap:.3f}")

    summary_class = 'a-ok' if is_fair else 'a-warn'
    st.markdown(
        f'<div class="{summary_class}"><strong>Audit summary</strong><br/>'
        f'Top review target: {html.escape(str(top_priority["Attribute"]))} · '
        f'Risk: {html.escape(str(top_priority["Risk"]))} · '
        f'Worst group: {html.escape(str(top_priority["Worst Group"]))}</div>',
        unsafe_allow_html=True,
    )

    audit_df = pd.DataFrame(audit_rows)
    display_audit_df = audit_df.copy()
    for column in ['DP Ratio', 'Disparate Impact', 'Accuracy Gap', 'F1 Gap']:
        display_audit_df[column] = display_audit_df[column].map(lambda value: f'{float(value):.3f}')

    st.markdown("#### Fairness Audit Matrix")
    st.dataframe(display_audit_df, width="stretch", hide_index=True)

    try:
        import plotly.graph_objects as go

        risk_fig = go.Figure(go.Scatter(
            x=audit_df['DP Ratio'],
            y=audit_df['F1 Gap'],
            mode='markers+text',
            text=audit_df['Attribute'],
            textposition='top center',
            marker=dict(
                size=[max(14, 20 + gap * 60) for gap in audit_df['Accuracy Gap']],
                color=['#22c55e' if status == 'Pass' else '#ef4444' for status in audit_df['Status']],
                opacity=0.85,
                line=dict(color='#f5f5f5', width=1),
            ),
        ))
        risk_fig.update_layout(
            title_text='Fairness risk map',
            xaxis_title='DP Ratio (closer to 1 is better)',
            yaxis_title='F1 Gap (lower is better)',
            height=340,
        )
        _plotly_chart(st, risk_fig)
    except Exception as e:
        _handle_action_error(st, 'Fairness risk map', e)

    selected_attr = st.selectbox(
        'Protected attribute drill-down',
        audit_df['Attribute'].tolist(),
        key='fairness_attribute_focus',
    )
    metrics = metrics_by_attribute[selected_attr]
    risk_label, risk_tone = _classify_fairness_risk(metrics)

    st.markdown(f"#### {selected_attr} Drill-down")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("DP Ratio", f"{metrics.get('demographic_parity_ratio', 0):.3f}")
    d2.metric("Disparate Impact", f"{metrics.get('disparate_impact', 0):.3f}")
    d3.metric("Accuracy Gap", f"{metrics.get('accuracy_gap', 0):.3f}")
    d4.metric("F1 Gap", f"{metrics.get('f1_gap', 0):.3f}")

    summary_class = 'a-ok' if risk_tone == 'ok' else 'a-info' if risk_tone == 'info' else 'a-warn'
    st.markdown(
        f'<div class="{summary_class}"><strong>{html.escape(selected_attr)}</strong><br/>'
        f'Risk level: {html.escape(risk_label)} · '
        f'Best group: {html.escape(str(metrics.get("best_group_by_f1") or "—"))} · '
        f'Worst group: {html.escape(str(metrics.get("worst_group_by_f1") or "—"))} · '
        f'Threshold: {float(metrics.get("threshold", active_threshold)):.2f}</div>',
        unsafe_allow_html=True,
    )

    group_rows = _build_group_metric_rows(metrics.get('group_metrics', {}))
    if group_rows:
        group_df, display_group_df = _format_group_metric_display(group_rows)
        st.dataframe(display_group_df, width="stretch", hide_index=True)

        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=group_df['Бүлэг'],
                y=group_df['Accuracy'],
                name='Accuracy',
                marker_color='#3b82f6',
                text=[f'{value:.2%}' for value in group_df['Accuracy']],
                textposition='auto',
            ))
            fig.add_trace(go.Bar(
                x=group_df['Бүлэг'],
                y=group_df['F1'],
                name='F1',
                marker_color='#22c55e',
                text=[f'{value:.2%}' for value in group_df['F1']],
                textposition='auto',
            ))
            fig.add_trace(go.Bar(
                x=group_df['Бүлэг'],
                y=group_df['Эерэг %'],
                name='Positive rate',
                marker_color='#eab308',
                text=[f'{value:.2%}' for value in group_df['Эерэг %']],
                textposition='auto',
            ))
            fig.update_layout(
                title_text=f'{selected_attr} group comparison',
                barmode='group',
                yaxis_range=[0, 1],
                height=360,
            )
            _plotly_chart(st, fig)
        except Exception as e:
            _handle_action_error(st, 'Fairness group comparison chart', e)

    recs = fr.get('recommendations', [])
    if recs:
        tone = 'ok' if is_fair else 'warn'
        _render_bullet_callout(st, 'Mitigation guidance', recs[:7], tone=tone)


# ============================================================================
# Legacy helper classes (backward compatibility)
# ============================================================================
class SidebarComponent:
    @staticmethod
    def render():
        render_sidebar()

class MetricsComponent:
    @staticmethod
    def render(metrics):
        import streamlit as st
        cols = st.columns(len(metrics))
        for col, (n, v) in zip(cols, metrics.items()):
            col.metric(n, f"{v:.4f}")

class ExplanationComponent:
    @staticmethod
    def render(explanation):
        import streamlit as st
        st.markdown(f"**Суурь:** {explanation.get('base_value', 'N/A')}")
        for it in explanation.get('contributions', [])[:10]:
            d = ICONS['up'] if it['shap_value'] > 0 else ICONS['down']
            st.markdown(f"{d} **{it['feature']}**: {it['shap_value']:.4f}")


def get_status_html(is_active, active_text, inactive_text):
    if is_active:
        return f'<span class="st-dot st-on"></span> {active_text}'
    return f'<span class="st-dot st-off"></span> {inactive_text}'
