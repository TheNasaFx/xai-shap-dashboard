"""
Dashboard App — XAI-SHAP Deep Dark Dashboard
=============================================
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO)

from typing import Any, Optional
import numpy as np
from src.dashboard.state import ensure_dashboard_state, get_workflow_status

logger = logging.getLogger(__name__)

# ============================================================================
# Deep Dark Professional Theme
# ============================================================================
PROFESSIONAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-0: #0a0a0a;
    --bg-1: #111111;
    --bg-2: #161616;
    --bg-3: #1e1e1e;
    --border: #222222;
    --text-1: #f5f5f5;
    --text-2: #b0b0b0;
    --text-3: #777777;
    --accent: #3b82f6;
    --accent-2: #8b5cf6;
    --success: #22c55e;
    --warning: #eab308;
    --error: #ef4444;
}

/* Global */
html, body, .main, [data-testid="stAppViewContainer"],
[data-testid="stApp"], [data-testid="stHeader"] {
    background-color: var(--bg-0) !important;
    color: var(--text-1);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.main { padding: 0 1rem; }
.block-container { max-width: 1200px; padding: 1.5rem 1rem; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Title */
.dash-title {
    font-size: 1.75rem; font-weight: 800; color: var(--text-1);
    letter-spacing: -0.04em; margin: 0; line-height: 1.3;
}
.dash-sub {
    font-size: 0.875rem; color: var(--text-2);
    margin: 0.25rem 0 1.5rem 0; font-weight: 400;
}

/* Workflow Summary */
.workflow-strip {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin: 0 0 1rem 0;
}
.workflow-card {
    background: linear-gradient(180deg, rgba(24,24,24,0.96), rgba(15,15,15,0.98));
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}
.workflow-kicker {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-3);
    margin-bottom: 0.35rem;
}
.workflow-value {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-1);
    line-height: 1.35;
}
.workflow-meta {
    font-size: 0.78rem;
    color: var(--text-2);
    margin-top: 0.4rem;
    line-height: 1.45;
}
.workflow-steps {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 0.25rem 0 0.9rem 0;
}
.workflow-step {
    border: 1px solid var(--border);
    background: var(--bg-1);
    border-radius: 999px;
    padding: 0.45rem 0.8rem;
    font-size: 0.78rem;
    color: var(--text-2);
}
.workflow-step.is-done {
    background: rgba(34,197,94,0.12);
    border-color: rgba(34,197,94,0.3);
    color: #d6ffe4;
}
.workflow-step.is-active {
    background: rgba(59,130,246,0.12);
    border-color: rgba(59,130,246,0.3);
    color: #dbe8ff;
}
.workflow-note {
    background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(139,92,246,0.08));
    border: 1px solid rgba(96,165,250,0.2);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin: 0 0 1.4rem 0;
    color: #dbe5f0;
    font-size: 0.86rem;
    line-height: 1.55;
}
@media (max-width: 992px) {
    .workflow-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px; background: var(--bg-1); padding: 4px;
    border-radius: 10px; border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    padding: 10px 20px; border-radius: 8px; font-weight: 500;
    font-size: 0.85rem; color: var(--text-2); transition: all 0.15s;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text-1); background: var(--bg-2); }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
}

/* Charts */
.stPlotlyChart { width: 100% !important; border-radius: 12px; overflow: hidden; }

/* Buttons */
.stButton > button {
    border-radius: 8px; font-weight: 600;
    font-family: 'Inter', sans-serif; transition: all 0.15s ease;
    border: 1px solid var(--border);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(59,130,246,0.15);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
    border: none;
}

/* Metrics */
[data-testid="metric-container"] {
    background: var(--bg-2) !important; border: 1px solid var(--border);
    border-radius: 10px; padding: 0.875rem;
}
[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700; color: var(--text-1) !important; }
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; color: var(--text-2) !important; }
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-1) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif !important; }

/* Markdown text */
[data-testid="stMarkdownContainer"] p { color: #c8d0d8; }
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 { color: var(--text-1) !important; }
[data-testid="stMarkdownContainer"] strong { color: var(--text-1); }

/* Inputs */
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label,
[data-testid="stFileUploader"] label { color: var(--text-2) !important; font-size: 0.85rem; }

/* DataFrames */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Expanders */
.streamlit-expanderHeader { font-weight: 500; font-size: 0.9rem; color: var(--text-1) !important; }
details { background: var(--bg-2) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }

/* File Uploader */
[data-testid="stFileUploader"] { border-radius: 10px; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Toast */
[data-testid="stToast"] { background: var(--bg-3) !important; border: 1px solid var(--border); }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-0); }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #444; }

/* Responsive */
@media (max-width: 768px) {
    .dash-title { font-size: 1.25rem; }
    .block-container { padding: 1rem 0.5rem; }
    .stTabs [data-baseweb="tab"] { padding: 8px 12px; font-size: 0.75rem; }
    .workflow-strip { grid-template-columns: 1fr; }
}

/* Footer */
.ft {
    margin-top: 3rem; padding: 1.5rem; text-align: center;
    color: var(--text-3); border-top: 1px solid var(--border);
    font-size: 0.75rem;
}
.ft a { color: var(--accent); text-decoration: none; }
.ft a:hover { text-decoration: underline; }
</style>
"""


def _next_step_text(status) -> str:
    if status["uploaded_data"] is None:
        return "Эхлэл: dataset upload хийх эсвэл sample dataset сонгоно уу. Систем эхлээд өгөгдлийг тань audit хийж байж дараагийн analysis руу орно."
    if not status["data_loaded"]:
        return "Дараагийн алхам: target column болон protected attributes-аа сонгоод Өгөгдөл tab дээрх боловсруулах алхмыг дуусгана уу."
    if not status["model_trained"]:
        return "Дараагийн алхам: Загвар tab дээр baseline болон candidate model-уудаа сургаад SHAP-д ашиглах active model-оо сонгоно уу."
    if not status["explanations_generated"]:
        return "Дараагийн алхам: SHAP explanation үүсгээд дараа нь visualization, stability, error analysis, fairness хэсгүүд рүү шилжинэ үү."
    return "Analysis-ready. Одоо visualization, advanced analysis, fairness evaluation, report export-оо шууд үргэлжлүүлж болно."


def _render_workflow_header(st_module):
    status = get_workflow_status(st_module)
    steps = [
        ("1. Өгөгдөл", status["data_loaded"]),
        ("2. Загвар", status["model_trained"]),
        ("3. SHAP", status["explanations_generated"]),
    ]

    first_pending = next((index for index, (_, done) in enumerate(steps) if not done), None)
    step_html = []
    for index, (label, done) in enumerate(steps):
        css_class = "workflow-step is-done" if done else "workflow-step is-active" if first_pending == index else "workflow-step"
        step_html.append(f'<div class="{css_class}">{label}</div>')

    dataset_value = status["dataset_name"] or "Dataset сонгоогүй"
    dataset_meta = (
        f'{status["row_count"]:,} мөр · {status["column_count"]} багана'
        if status["uploaded_data"] is not None
        else "CSV upload эсвэл sample dataset"
    )
    target_value = status["target_col"] or "Target сонгоогүй"
    target_meta = "Preprocessing бэлэн" if status["data_loaded"] else "Target болон preprocessing pending"
    model_value = status["selected_model_name"] or "Model сонгоогүй"
    model_meta = (
        f'{status["trained_model_count"]} trained model candidate'
        if status["trained_model_count"]
        else "Training pending"
    )
    shap_value = "Бэлэн" if status["explanations_generated"] else "Хүлээгдэж байна"
    shap_meta = (
        "Visualization ба advanced analysis ашиглахад бэлэн"
        if status["explanations_generated"]
        else f'{status["completed_steps"]}/3 core stages complete'
    )

    st_module.markdown(
        f"""
        <div class="workflow-steps">{''.join(step_html)}</div>
        <div class="workflow-strip">
            <div class="workflow-card">
                <div class="workflow-kicker">Current Dataset</div>
                <div class="workflow-value">{dataset_value}</div>
                <div class="workflow-meta">{dataset_meta}</div>
            </div>
            <div class="workflow-card">
                <div class="workflow-kicker">Current Target</div>
                <div class="workflow-value">{target_value}</div>
                <div class="workflow-meta">{target_meta}</div>
            </div>
            <div class="workflow-card">
                <div class="workflow-kicker">Active Model</div>
                <div class="workflow-value">{model_value}</div>
                <div class="workflow-meta">{model_meta}</div>
            </div>
            <div class="workflow-card">
                <div class="workflow-kicker">Explanation Status</div>
                <div class="workflow-value">{shap_value}</div>
                <div class="workflow-meta">{shap_meta}</div>
            </div>
        </div>
        <div class="workflow-note"><strong>Next step:</strong> {_next_step_text(status)}</div>
        """,
        unsafe_allow_html=True,
    )


def main(framework=None):
    """Dashboard entry point."""
    import streamlit as st
    from src.dashboard.components import (
        inject_custom_css,
        render_sidebar,
        render_data_section,
        render_model_section,
        render_explanation_section,
        render_visualization_section,
        render_fairness_section,
        render_data_quality_section,
        render_model_metrics_section,
        render_advanced_analysis_section,
    )

    st.set_page_config(
        page_title="XAI-SHAP Dashboard",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject both CSS systems
    st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)
    inject_custom_css()

    # Title
    st.markdown("""
    <div class="dash-title">XAI-SHAP Dashboard</div>
    <div class="dash-sub">Тайлбарлах боломжтой AI · SHAP Framework</div>
    """, unsafe_allow_html=True)

    # Session state
    ensure_dashboard_state(st, framework=framework)
    _render_workflow_header(st)

    # Sidebar
    with st.sidebar:
        render_sidebar()

    # 8 Tabs — clean names, no emoji
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Өгөгдөл", "Чанар", "Загвар", "Гүйцэтгэл",
        "Тайлбарууд", "Визуал", "Шинжилгээ", "Шударга",
    ])

    with tab1: render_data_section()
    with tab2: render_data_quality_section()
    with tab3: render_model_section()
    with tab4: render_model_metrics_section()
    with tab5: render_explanation_section()
    with tab6: render_visualization_section()
    with tab7: render_advanced_analysis_section()
    with tab8: render_fairness_section()

    # Footer
    st.markdown('<div class="ft">XAI-SHAP Framework v2.0 · Дипломын ажил · 2026</div>', unsafe_allow_html=True)


def create_dashboard(framework=None):
    """Legacy compatibility."""
    import streamlit as st
    st.set_page_config(page_title="XAI-SHAP", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)
    ensure_dashboard_state(st, framework=framework)
    return st


def run_dashboard(framework=None, debug: bool = False):
    """Legacy compatibility."""
    main(framework=framework)


if __name__ == "__main__":
    main()
