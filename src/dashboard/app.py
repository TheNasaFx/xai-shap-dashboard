"""
Dashboard App - Streamlit Dashboard-ийн Үндсэн Програм
======================================================

XAI-SHAP framework-д зориулсан интерактив визуал аналитик dashboard.
Загварын тайлбаруудыг судлах хэрэглэгчдэд ээлтэй интерфейсийг хангана.

Зохиогч: XAI-SHAP Framework
"""

import sys
from pathlib import Path

# Import-уудад зориулж project root-ийг path-д нэмэх
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO)

from typing import Any, Optional
import numpy as np

logger = logging.getLogger(__name__)
logger.info(f"App.py started, Python {sys.version}, project_root={project_root}")


# ============================================================================
# Professional Design System
# ============================================================================

# Unicode icons (no emojis)
ICONS = {
    "logo": "◈",
    "data": "◈",
    "model": "◆", 
    "explain": "◇",
    "chart": "▣",
    "fairness": "◎",
}

# Professional CSS stylesheet
PROFESSIONAL_CSS = """
<style>
/* ============================================
   XAI-SHAP Professional Dashboard Theme
   ============================================ */

/* Variables */
:root {
    --primary-color: #3b82f6;
    --primary-dark: #2563eb;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --error-color: #ef4444;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --bg-primary: #ffffff;
    --bg-secondary: #f8fafc;
    --border-color: #e2e8f0;
}

/* Main Container */
.main {
    padding: 0rem 1rem;
    background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
}

/* Dashboard Title */
.dashboard-title {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 4px;
    letter-spacing: -0.025em;
}

.dashboard-subtitle {
    font-size: 1rem;
    color: var(--text-secondary);
    margin-bottom: 24px;
}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: var(--bg-secondary);
    padding: 8px;
    border-radius: 12px;
    border: 1px solid var(--border-color);
}

.stTabs [data-baseweb="tab"] {
    padding: 12px 24px;
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(59, 130, 246, 0.1);
}

.stTabs [aria-selected="true"] {
    background: var(--primary-color) !important;
    color: white !important;
}

/* Plotly Charts */
.stPlotlyChart {
    width: 100%;
    border-radius: 12px;
    overflow: hidden;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
}

/* Metrics */
[data-testid="metric-container"] {
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-primary);
}

[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: var(--text-secondary);
    font-size: 0.875rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
    border-right: 1px solid var(--border-color);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text-primary);
}

/* Expanders */
.streamlit-expanderHeader {
    font-weight: 500;
    color: var(--text-primary);
}

/* DataFrames */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border-color);
}

/* File Uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border-color);
    border-radius: 12px;
    padding: 20px;
    transition: all 0.2s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--primary-color);
    background: rgba(59, 130, 246, 0.05);
}

/* Success/Warning/Error messages */
.stSuccess {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border: 1px solid #6ee7b7;
    border-radius: 8px;
}

.stWarning {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border: 1px solid #fcd34d;
    border-radius: 8px;
}

.stError {
    background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
    border: 1px solid #fca5a5;
    border-radius: 8px;
}

/* Slider */
.stSlider > div > div > div > div {
    background: var(--primary-color);
}

/* Selectbox */
.stSelectbox > div > div {
    border-radius: 8px;
}

/* Footer */
.footer {
    margin-top: 40px;
    padding: 20px;
    text-align: center;
    color: var(--text-secondary);
    border-top: 1px solid var(--border-color);
    font-size: 0.875rem;
}
</style>
"""


def create_dashboard(framework=None):
    """
    Streamlit dashboard-ийг үүсгэж тохируулах.
    
    Параметрүүд:
        framework: XAIFramework instance (заавал биш)
        
    Буцаах:
        Streamlit app-ийн тохиргоо
    """
    import streamlit as st
    
    # Хуудасны тохиргоо - emoji-гүй, мэргэжлийн
    st.set_page_config(
        page_title="XAI-SHAP Визуал Аналитик",
        page_icon="◈",  # Unicode icon instead of emoji
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Мэргэжлийн CSS оруулах
    st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)
    
    # Гарчиг - emoji-гүй
    st.markdown(f"""
    <div class="dashboard-title">{ICONS['logo']} XAI-SHAP Визуал Аналитик</div>
    <div class="dashboard-subtitle">Хар хайрцагт AI загваруудыг тайлбарлах интерактив платформ</div>
    """, unsafe_allow_html=True)
    
    # Session state эхлүүлэх
    if 'framework' not in st.session_state:
        st.session_state.framework = framework
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'explanations_generated' not in st.session_state:
        st.session_state.explanations_generated = False
    
    return st


def run_dashboard(
    framework=None,
    debug: bool = False
):
    """
    Dashboard-ийг ажиллуулах.
    
    Параметрүүд:
        framework: XAIFramework instance
        debug: Debug горим идэвхижүүлэх эсэх
    """
    import streamlit as st
    from src.dashboard.components import (
        render_sidebar,
        render_data_section,
        render_model_section,
        render_explanation_section,
        render_visualization_section,
        render_fairness_section
    )
    
    # Dashboard эхлүүлэх
    create_dashboard(framework)
    
    # Framework тохируулах
    if 'framework' not in st.session_state or st.session_state.framework is None:
        from src.core.framework import XAIFramework
        st.session_state.framework = XAIFramework()
    
    # Хажуугийн самбар
    with st.sidebar:
        render_sidebar()
    
    # Үндсэн агуулга - emoji-гүй, цэвэр текст
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        f"{ICONS['data']} Өгөгдөл",
        f"{ICONS['model']} Загвар", 
        f"{ICONS['explain']} Тайлбарууд",
        f"{ICONS['chart']} Визуал",
        f"{ICONS['fairness']} Шударга Байдал"
    ])
    
    with tab1:
        render_data_section()
    with tab2:
        render_model_section()
    with tab3:
        render_explanation_section()
    with tab4:
        render_visualization_section()
    with tab5:
        render_fairness_section()
    
    # Footer
    st.markdown("""
    <div class="footer">
        XAI-SHAP Framework | 
        <a href="https://github.com" target="_blank">Documentation</a> | 
        <a href="https://github.com" target="_blank">Report Issue</a>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Dashboard-ийн үндсэн эхлэх цэг."""
    import streamlit as st
    from src.dashboard.components import (
        render_sidebar,
        render_data_section,
        render_model_section,
        render_explanation_section,
        render_visualization_section,
        render_fairness_section
    )
    
    # Хуудасны тохиргоо - мэргэжлийн дизайн
    st.set_page_config(
        page_title="XAI-SHAP Визуал Аналитик",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Мэргэжлийн CSS
    st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)
    
    # Гарчиг
    st.markdown(f"""
    <div class="dashboard-title">{ICONS['logo']} XAI-SHAP Визуал Аналитик Dashboard</div>
    <div class="dashboard-subtitle">SHAP ашиглан Тайлбарлах боломжтой AI-д зориулсан Framework</div>
    """, unsafe_allow_html=True)
    
    # Session state эхлүүлэх
    if 'framework' not in st.session_state:
        from src.core.framework import XAIFramework
        st.session_state.framework = XAIFramework()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'explanations_generated' not in st.session_state:
        st.session_state.explanations_generated = False
    
    # Хажуугийн самбар
    with st.sidebar:
        render_sidebar()
    
    # Үндсэн агуулга - мэргэжлийн табууд
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        f"{ICONS['data']} Өгөгдөл",
        f"{ICONS['model']} Загвар",
        f"{ICONS['explain']} Тайлбарууд",
        f"{ICONS['chart']} Визуализациуд",
        f"{ICONS['fairness']} Шударга Байдал"
    ])
    
    with tab1:
        render_data_section()
    with tab2:
        render_model_section()
    with tab3:
        render_explanation_section()
    with tab4:
        render_visualization_section()
    with tab5:
        render_fairness_section()
    
    # Footer
    st.markdown("""
    <div class="footer">
        XAI-SHAP Framework | 
        <a href="https://github.com" target="_blank">Баримт бичиг</a> | 
        <a href="https://github.com" target="_blank">Асуудал мэдэгдэх</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
