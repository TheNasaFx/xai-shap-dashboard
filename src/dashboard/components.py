"""
Dashboard Components - Мэргэжлийн UI бүрэлдэхүүнүүд
====================================================

Streamlit dashboard-д зориулсан дахин ашиглах боломжтой модуль бүрэлдэхүүнүүд.
Мэргэжлийн дизайн, session state удирдлагатай.

Зохиогч: XAI-SHAP Framework
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Professional Icon System (Unicode symbols - no emojis)
# ============================================================================
ICONS = {
    "check": "●",       # Filled circle - success/active
    "pending": "○",     # Empty circle - pending/inactive 
    "data": "◈",        # Data/database
    "model": "◆",       # Model/AI
    "explain": "◇",     # Explanation
    "chart": "▣",       # Visualization
    "fairness": "◎",    # Fairness/balance
    "settings": "⚙",    # Settings gear
    "arrow": "→",       # Arrow right
    "bullet": "•",      # Bullet point
    "refresh": "↻",     # Refresh/reload
    "upload": "↑",      # Upload
    "download": "↓",    # Download
    "play": "▶",        # Play/run
    "target": "◉",      # Target/goal
    "info": "ℹ",        # Information
    "warning": "⚠",     # Warning
    "up": "▲",          # Up arrow
    "down": "▼",        # Down arrow
}


# ============================================================================
# Custom CSS Styles
# ============================================================================
CUSTOM_CSS = """
<style>
/* Global Styles */
.main { padding: 0rem 1rem; }
.stPlotlyChart { width: 100%; }

/* Status Card */
.status-card {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
}

.status-item {
    margin-bottom: 12px;
}

.status-item:last-child {
    margin-bottom: 0;
}

.status-label {
    font-size: 13px;
    color: #64748b;
    font-weight: 500;
    margin-bottom: 4px;
}

.status-active {
    color: #10b981;
    font-weight: 600;
}

.status-inactive {
    color: #94a3b8;
}

/* Section Headers */
.section-header {
    color: #1e293b;
    font-weight: 600;
    font-size: 1.5rem;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid #e2e8f0;
}

/* Info Box */
.info-box {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border: 1px solid #93c5fd;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 14px;
    color: #1e40af;
}

/* Success Box */
.success-box {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border: 1px solid #6ee7b7;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
}

/* Warning Box */
.warning-box {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border: 1px solid #fcd34d;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
}

/* Metric Card */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1e293b;
}

.metric-label {
    font-size: 0.875rem;
    color: #64748b;
    margin-top: 4px;
}

/* Help Text */
.help-text {
    font-size: 12px;
    color: #64748b;
    font-style: italic;
    margin-top: 4px;
}

/* Feature List */
.feature-positive {
    color: #059669;
}

.feature-negative {
    color: #dc2626;
}
</style>
"""


def inject_custom_css():
    """Inject custom CSS into the app."""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _format_dtype(dtype) -> str:
    """Numpy dtype-ийг хэрэглэгчдэд ойлгомжтой нэр рүү хөрвүүлэх."""
    dtype_str = str(dtype)
    dtype_map = {
        'int64': 'Бүхэл тоо (int64)',
        'int32': 'Бүхэл тоо (int32)',
        'float64': 'Бутархай тоо (float64)',
        'float32': 'Бутархай тоо (float32)',
        'object': 'Текст (object)',
        'bool': 'Тийм/Үгүй (bool)',
        'datetime64[ns]': 'Огноо/Цаг',
        'category': 'Ангилал (category)',
    }
    return dtype_map.get(dtype_str, dtype_str)


def _format_value(value) -> str:
    """Numpy утгыг ойлгомжтой Python утга руу хөрвүүлэх (np.int64 → int)."""
    if isinstance(value, (np.integer,)):
        return str(int(value))
    elif isinstance(value, (np.floating,)):
        return str(float(value))
    elif isinstance(value, np.bool_):
        return str(bool(value))
    return str(value)


def get_status_html(is_active: bool, active_text: str, inactive_text: str) -> str:
    """Generate status indicator HTML."""
    if is_active:
        return f'<span class="status-active">{ICONS["check"]} {active_text}</span>'
    return f'<span class="status-inactive">{ICONS["pending"]} {inactive_text}</span>'


def render_sidebar():
    """Хажуугийн самбарыг удирдлагуудтай харуулах."""
    import streamlit as st
    
    st.header(f"{ICONS['settings']} Удирдлагууд")
    
    # Framework-ийн төлөв шалгаж session state-г синхрончлох
    st.subheader("Системийн Төлөв")
    
    framework = st.session_state.get('framework')
    
    # Framework state-аас session state-г синхрончлох
    data_loaded = st.session_state.get('data_loaded', False)
    model_trained = st.session_state.get('model_trained', False)
    shap_generated = st.session_state.get('explanations_generated', False)
    
    # Framework дээр өгөгдөл/загвар байвал session state-г шинэчлэх
    if framework is not None:
        if framework.X_train is not None and not data_loaded:
            st.session_state['data_loaded'] = True
            data_loaded = True
        if framework.model is not None and not model_trained:
            st.session_state['model_trained'] = True
            model_trained = True
        if framework.shap_values is not None and not shap_generated:
            st.session_state['explanations_generated'] = True
            shap_generated = True
    
    # Status card HTML
    status_html = f"""
    <div class="status-card">
        <div class="status-item">
            <div class="status-label">{ICONS['data']} Өгөгдөл</div>
            {get_status_html(data_loaded, "Ачаалагдсан", "Ачаалаагүй")}
        </div>
        <div class="status-item">
            <div class="status-label">{ICONS['model']} Загвар</div>
            {get_status_html(model_trained, "Сургагдсан", "Сургаагүй")}
        </div>
        <div class="status-item">
            <div class="status-label">{ICONS['explain']} SHAP</div>
            {get_status_html(shap_generated, "Үүсгэгдсэн", "Үүсгэгдээгүй")}
        </div>
    </div>
    """
    st.markdown(status_html, unsafe_allow_html=True)
    
    st.divider()
    
    # Түргэн үйлдлүүд
    st.subheader("Түргэн Үйлдлүүд")
    
    if st.button(f"{ICONS['refresh']} Бүгдийг Дахин Тохируулах", width='stretch'):
        # Clear all session state except framework instance
        keys_to_clear = ['data_loaded', 'model_trained', 'explanations_generated', 
                         'uploaded_data', 'target_col', 'explanations']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        # Reset framework
        from src.core.framework import XAIFramework
        st.session_state.framework = XAIFramework()
        st.rerun()
    
    # Экспорт сонголтууд
    st.subheader("Тайлан Экспорт")
    
    export_format = st.selectbox(
        "Формат сонгох",
        ["HTML", "PDF", "JSON"],
        key="export_format",
        help="Тайлангийн форматыг сонгоно уу"
    )
    
    if st.button(f"{ICONS['download']} Тайлан Татах", width='stretch'):
        st.info(f"{ICONS['info']} Тайлан экспортлох функц - тун удахгүй")
    
    st.divider()
    
    # Тухай хэсэг
    st.subheader("Тухай")
    st.markdown(f"""
    **XAI-SHAP Framework**
    
    SHAP утгууд ашиглан тайлбарлах боломжтой AI-д зориулсан визуал аналитик систем.
    
    {ICONS['bullet']} Интерактив визуализациуд
    
    {ICONS['bullet']} Локал болон глобал тайлбарууд
    
    {ICONS['bullet']} Шударга байдлын шинжилгээ
    """)


def render_data_section():
    """Өгөгдлийн тойм хэсгийг харуулах."""
    import streamlit as st
    
    st.markdown(f"""
    <div class="section-header">{ICONS['data']} Өгөгдлийн Тойм</div>
    """, unsafe_allow_html=True)
    
    framework = st.session_state.get('framework')
    
    # Өгөгдөл аль хэдийн боловсруулагдсан бол мэдээлэл харуулах
    data_loaded = st.session_state.get('data_loaded', False)
    if data_loaded and framework is not None and framework.X_train is not None:
        st.markdown(f"""
        <div class="success-box">
            <strong>{ICONS['check']} Өгөгдөл Боловсруулагдсан</strong><br/>
            {ICONS['bullet']} Сургалтын дээжүүд: {len(framework.X_train):,}<br/>
            {ICONS['bullet']} Тест дээжүүд: {len(framework.X_test):,}<br/>
            {ICONS['bullet']} Шинж чанарууд: {len(framework.feature_names)}
        </div>
        """, unsafe_allow_html=True)
    
    # Файл байршуулах
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "CSV файл байршуулах",
            type=['csv'],
            help="Өгөгдлийн багцаа CSV форматаар байршуулна уу. UTF-8 кодчилолтой файл шаардлагатай."
        )
    
    with col2:
        st.markdown("**Эсвэл жишээ өгөгдөл ашиглах:**")
        use_sample = st.button(f"{ICONS['play']} Жишээ Өгөгдөл Ачаалах", width='stretch')
    
    if use_sample:
        # Breast Cancer жишээ өгөгдлийн багц ачаалах
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['diagnosis'] = data.target  # 0=malignant, 1=benign
        st.session_state['uploaded_data'] = df
        
        # Шинэ өгөгдөл ачаалсан тул хуучин state-үүдийг цэвэрлэх
        st.session_state['data_loaded'] = False
        st.session_state['model_trained'] = False
        st.session_state['explanations_generated'] = False
        
        st.success(f"{ICONS['check']} Breast Cancer жишээ өгөгдлийн багц ачаалагдлаа! (569 дээж, 30 шинж чанар)")
        st.rerun()  # UI шинэчлэх
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['uploaded_data'] = df
        
        # Шинэ файл ачаалсан тул хуучин state-үүдийг цэвэрлэх
        if st.session_state.get('model_trained', False):
            st.session_state['data_loaded'] = False
            st.session_state['model_trained'] = False
            st.session_state['explanations_generated'] = False
        st.success(f"{ICONS['check']} Өгөгдлийн багц ачаалагдлаа: {df.shape[0]} мөр, {df.shape[1]} багана")
    
    # Өгөгдөл харуулах
    if 'uploaded_data' in st.session_state:
        df = st.session_state['uploaded_data']
        
        st.subheader("Өгөгдлийн Урьдчилсан Харагдац")
        st.dataframe(df.head(10), width='stretch')
        
        # Өгөгдлийн статистик
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Нийт Мөр", f"{df.shape[0]:,}")
        with col2:
            st.metric("Баганууд", df.shape[1])
        with col3:
            missing = df.isnull().sum().sum()
            st.metric("Дутуу Утга", missing)
        with col4:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Тоон Багана", len(numeric_cols))
        
        # Баганын мэдээлэл
        with st.expander(f"{ICONS['info']} Баганын Дэлгэрэнгүй Мэдээлэл", expanded=False):
            col_info = pd.DataFrame({
                'Багана': df.columns.tolist(),
                'Төрөл': [_format_dtype(dtype) for dtype in df.dtypes.values],
                'Null Биш': df.count().values.tolist(),
                'Өвөрмөц Утга': df.nunique().values.tolist(),
                'Дутуу %': [f"{100*df[col].isnull().sum()/len(df):.1f}%" for col in df.columns]
            })
            st.dataframe(col_info, width='stretch')
        
        # Зорилтот багана сонгох
        st.subheader(f"{ICONS['target']} Өгөгдлийн Багцыг Тохируулах")
        
        # Тайлбар текст нэмэх
        st.markdown("""
        <div class="info-box">
            <strong>Зорилтот багана</strong> гэдэг нь таны загвар таамаглахыг хүсэж буй үр дүнгийн багана юм. 
            Жишээ нь: "price" (үнэ), "label" (шошго), "outcome" (үр дүн) гэх мэт.
            Ихэвчлэн сүүлийн багана зорилтот багана байдаг.
        </div>
        """, unsafe_allow_html=True)
        
        target_col = st.selectbox(
            "Зорилтот Багана Сонгох",
            df.columns.tolist(),
            index=len(df.columns) - 1,
            help="Загвараар таамаглахыг хүсэж буй баганаа сонгоно уу. Энэ нь ихэвчлэн 'target', 'label', 'y' гэх мэт нэртэй байдаг."
        )
        
        # Сонгосон зорилтот баганын мэдээлэл харуулах
        if target_col:
            target_info = df[target_col]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Сонгосон зорилт:** `{target_col}`")
                st.markdown(f"**Төрөл:** {_format_dtype(target_info.dtype)}")
            with col2:
                unique_count = target_info.nunique()
                st.markdown(f"**Өвөрмөц утгууд:** {unique_count}")
                if unique_count <= 10:
                    # numpy төрлүүдийг Python төрөл рүү хөрвүүлж харуулах
                    clean_values = [_format_value(v) for v in target_info.unique()[:10]]
                    st.markdown(f"**Утгууд:** {clean_values}")
        
        protected_attrs = st.multiselect(
            "Хамгаалагдсан Атрибутууд (заавал биш)",
            [c for c in df.columns if c != target_col],
            help="Шударга байдлын шинжилгээнд ашиглах баганууд. Жишээ нь: 'gender', 'age', 'race' гэх мэт хүн ам зүйн баганууд."
        )
        
        # Боловсруулах товч
        if st.button(f"{ICONS['play']} Өгөгдөл Боловсруулах", type="primary", width='stretch'):
            with st.spinner("Өгөгдөл боловсруулж байна..."):
                try:
                    framework.load_data(
                        df, target=target_col,
                        protected_attributes=protected_attrs
                    )
                    st.session_state['target_col'] = target_col
                    st.session_state['data_loaded'] = True
                    
                    # Шинэ өгөгдөл ачаалсан тул хуучин загвар болон тайлбаруудыг цэвэрлэх
                    # Энэ нь feature тооны зөрүүтэй алдаанаас сэргийлнэ
                    st.session_state['model_trained'] = False
                    st.session_state['explanations_generated'] = False
                    if 'explanations' in st.session_state:
                        del st.session_state['explanations']
                    
                    # Амжилттай мэдээллийг session state-д хадгалах (rerun-ий дараа харуулахын тулд)
                    st.session_state['data_success_message'] = {
                        'train_samples': len(framework.X_train),
                        'test_samples': len(framework.X_test),
                        'features': len(framework.feature_names)
                    }
                    
                    # Toast мэдэгдэл харуулах (rerun-д хадгалагдана)
                    st.toast(f"{ICONS['check']} Өгөгдөл амжилттай боловсруулагдлаа!", icon="✅")
                    
                    # Session state шинэчлэгдсэн тул UI шинэчлэх
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"{ICONS['warning']} Өгөгдөл боловсруулахад алдаа: {e}")
        
        # Амжилттай мэдээллийг харуулах (rerun-ий дараа)
        if 'data_success_message' in st.session_state:
            msg = st.session_state['data_success_message']
            st.success(f"{ICONS['check']} Өгөгдөл амжилттай боловсруулагдлаа!")
            st.markdown(f"""
            <div class="success-box">
                <strong>Боловсруулалт Дууссан:</strong><br/>
                {ICONS['bullet']} Сургалтын дээжүүд: {msg['train_samples']:,}<br/>
                {ICONS['bullet']} Тест дээжүүд: {msg['test_samples']:,}<br/>
                {ICONS['bullet']} Шинж чанарууд: {msg['features']}
            </div>
            """, unsafe_allow_html=True)
            # Мэдээллийг нэг удаа харуулсны дараа устгах
            del st.session_state['data_success_message']


def render_model_section():
    """Загвар сургах хэсгийг харуулах."""
    import streamlit as st
    
    st.markdown(f"""
    <div class="section-header">{ICONS['model']} Загвар Сургах</div>
    """, unsafe_allow_html=True)
    
    framework = st.session_state.get('framework')
    
    # Session state болон framework хоёуланг нь шалгах
    # Framework дээр өгөгдөл ачаалагдсан бол session state-г синхрончлох
    data_loaded = st.session_state.get('data_loaded', False)
    if framework is not None and framework.X_train is not None:
        if not data_loaded:
            st.session_state['data_loaded'] = True
            data_loaded = True
    
    if not data_loaded:
        st.markdown(f"""
        <div class="warning-box">
            {ICONS['warning']} <strong>Өгөгдөл шаардлагатай</strong><br/>
            Эхлээд "Өгөгдөл" хэсэгт очиж өгөгдөл ачаалж боловсруулна уу.
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown(f"""
    <div class="info-box">
        Энд та ML загвар сонгож, hyperparameter тохируулан сургах боломжтой.
        Загвар сургасны дараа <strong>Тайлбарууд</strong> болон <strong>Визуализациуд</strong> хэсэг идэвхжинэ.
    </div>
    """, unsafe_allow_html=True)
    
    # Загвар сонгох
    st.subheader("Загварын Тохиргоо")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Загварын Төрөл",
            [
                "xgboost", "lightgbm", "catboost",  # Gradient Boosting
                "random_forest", "extra_trees", "gradient_boosting", "adaboost",  # Tree Ensemble
                "neural_network",  # Neural Network
                "logistic_regression", "svm"  # Linear & Kernel
            ],
            help="Сургах загварын төрлийг сонгоно уу. XGBoost, LightGBM, CatBoost нь ихэвчлэн хамгийн сайн үр дүн өгдөг."
        )
    
    with col2:
        st.markdown("**Загварын Тайлбар:**")
        descriptions = {
            "xgboost": f"{ICONS['bullet']} <strong>XGBoost</strong> — Хурдан, нарийвчлалтай. Ихэнх тохиолдолд хамгийн сайн үр дүн өгдөг. Эхлэгчдэд тохиромжтой",
            "lightgbm": f"{ICONS['bullet']} <strong>LightGBM</strong> — Маш хурдан сургалт, том өгөгдөлд (100,000+ мөр) хамгийн тохиромжтой",
            "catboost": f"{ICONS['bullet']} <strong>CatBoost</strong> — Текст/ангилал төрлийн баганатай өгөгдөлд шилдэг, overfitting (хэт сургалт) багатай",
            "random_forest": f"{ICONS['bullet']} <strong>Random Forest</strong> — Тогтвортой, ойлгоход хялбар. Олон мод нэгтгэж таамаглал хийнэ",
            "extra_trees": f"{ICONS['bullet']} <strong>Extra Trees</strong> — Random Forest-тэй төстэй боловч илүү хурдан. Санамсаргүй хуваалт хийнэ",
            "gradient_boosting": f"{ICONS['bullet']} <strong>Gradient Boosting</strong> — Сонгодог sklearn boosting. Жижиг өгөгдөлд тогтвортой",
            "adaboost": f"{ICONS['bullet']} <strong>AdaBoost</strong> — Энгийн бөгөөд үр дүнтэй. Буруу таамагласан дээжүүдэд анхаарал хандуулна",
            "neural_network": f"{ICONS['bullet']} <strong>Neural Network (MLP)</strong> — Нарийн хэв маягийг сурна. Их өгөгдөл шаарддаг, /удаан/",
            "logistic_regression": f"{ICONS['bullet']} <strong>Logistic Regression</strong> — Шугаман загвар, хамгийн хялбар тайлбарлагдана. Шугаман хамаарал бүхий өгөгдөлд тохиромжтой",
            "svm": f"{ICONS['bullet']} <strong>SVM</strong> — Жижиг/дунд өгөгдөлд сайн. Kernel аргаар шугаман бус хамаарлыг олно"
        }
        st.markdown(f"""
        <div class="info-box">
            {descriptions.get(model_type, "")}
        </div>
        """, unsafe_allow_html=True)
    
    # Hyperparameter-ууд
    st.subheader("Hyperparameter-ууд")
    
    params = {}
    
    if model_type == "xgboost":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.slider("n_estimators", 50, 500, 100, 
                                     help="Модны тоо. Их байх тусам сайн боловч удаан.")
        with col2:
            max_depth = st.slider("max_depth", 2, 15, 6,
                                  help="Модны гүн. Их байх тусам нарийн загвар.")
        with col3:
            learning_rate = st.slider("learning_rate", 0.01, 0.3, 0.1,
                                      help="Сургалтын хурд. Бага байх тусам тогтвортой.")
        params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate}
    
    elif model_type == "lightgbm":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.slider("n_estimators", 50, 500, 100, 
                                     help="Boosting давталтын тоо")
        with col2:
            max_depth = st.slider("max_depth", 2, 15, 6,
                                  help="Модны хамгийн их гүн")
        with col3:
            learning_rate = st.slider("learning_rate", 0.01, 0.3, 0.1,
                                      help="Сургалтын хурд")
        params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate}
    
    elif model_type == "catboost":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.slider("iterations", 50, 500, 100, 
                                     help="Boosting давталтын тоо")
        with col2:
            max_depth = st.slider("depth", 2, 10, 6,
                                  help="Модны гүн (CatBoost-д 10 хүртэл)")
        with col3:
            learning_rate = st.slider("learning_rate", 0.01, 0.3, 0.1,
                                      help="Сургалтын хурд")
        params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate}
    
    elif model_type == "random_forest":
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("n_estimators", 50, 500, 100,
                                     help="Модны тоо")
        with col2:
            max_depth = st.slider("max_depth", 2, 20, 10,
                                  help="Модны хамгийн их гүн")
        params = {'n_estimators': n_estimators, 'max_depth': max_depth}
    
    elif model_type == "extra_trees":
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("n_estimators", 50, 500, 100,
                                     help="Модны тоо")
        with col2:
            max_depth = st.slider("max_depth", 2, 30, 15,
                                  help="Модны хамгийн их гүн (None=хязгааргүй)")
        params = {'n_estimators': n_estimators, 'max_depth': max_depth if max_depth < 30 else None}
    
    elif model_type == "gradient_boosting":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.slider("n_estimators", 50, 300, 100, 
                                     help="Boosting давталтын тоо")
        with col2:
            max_depth = st.slider("max_depth", 1, 10, 3,
                                  help="Модны гүн (3-5 ихэвчлэн сайн)")
        with col3:
            learning_rate = st.slider("learning_rate", 0.01, 0.3, 0.1,
                                      help="Сургалтын хурд")
        params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate}
    
    elif model_type == "adaboost":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.slider("n_estimators", 10, 200, 50, 
                                     help="Сул суралцагчдын тоо")
        with col2:
            max_depth = st.slider("base_max_depth", 1, 5, 3,
                                  help="Суурь модны гүн")
        with col3:
            learning_rate = st.slider("learning_rate", 0.1, 2.0, 1.0,
                                      help="Сургалтын хурд")
        params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate}
    
    elif model_type == "neural_network":
        col1, col2 = st.columns(2)
        with col1:
            hidden_layers = st.text_input("hidden_layers", "128,64,32",
                                          help="Нуугдмал давхаргуудын хэмжээ, таслалаар тусгаарлана")
        with col2:
            epochs = st.slider("epochs", 50, 300, 100,
                               help="Сургалтын давталтын тоо")
        params = {'hidden_layers': [int(x.strip()) for x in hidden_layers.split(',')], 'epochs': epochs}
    
    elif model_type == "logistic_regression":
        col1, col2 = st.columns(2)
        with col1:
            penalty = st.selectbox("penalty", ["l2", "l1", "elasticnet"],
                                   help="Regularization төрөл")
        with col2:
            C = st.slider("C", 0.01, 10.0, 1.0,
                          help="Урвуу regularization хүч (их=бага regularization)")
        params = {'penalty': penalty, 'C': C}
    
    elif model_type == "svm":
        col1, col2, col3 = st.columns(3)
        with col1:
            kernel = st.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"],
                                  help="Kernel функц")
        with col2:
            C = st.slider("C", 0.1, 10.0, 1.0,
                          help="Regularization параметр")
        with col3:
            gamma = st.selectbox("gamma", ["scale", "auto"],
                                 help="Kernel коэффициент")
        params = {'kernel': kernel, 'C': C, 'gamma': gamma}
    
    else:
        st.info(f"{ICONS['info']} Үндсэн тохиргоо ашиглагдана")
        params = {}
    
    # Том өгөгдлийн анхааруулга
    df = st.session_state.get('uploaded_data')
    if df is not None and len(df) >= 10000:
        st.markdown(f"""
        <div class="warning-box">
            {ICONS['warning']} <strong>Том өгөгдлийн багц ({len(df):,} мөр)</strong><br/>
            Энэ хэмжээний өгөгдөл дээр сургалт удаан үргэлжлэх магадлалтай. 
            <strong>Google Colab</strong>-ийн үнэгүй GPU ашиглан хурдан сургах боломжтой.
        </div>
        """, unsafe_allow_html=True)
    
    # Сургах товч (Локал + Cloud)
    st.markdown("")  # Spacer
    
    col_train1, col_train2 = st.columns(2)
    
    with col_train1:
        local_train = st.button(f"{ICONS['play']} Загвар Сургах (Локал)", type="primary", width='stretch')
    
    with col_train2:
        cloud_train = st.button(f"{ICONS['upload']} Google Colab-д Сургах", width='stretch',
                                help="Google Colab notebook үүсгэж татах. Үнэгүй GPU ашиглан том өгөгдөл дээр хурдан сургах боломжтой.")
    
    if cloud_train:
        if df is not None:
            with st.spinner("Google Colab notebook үүсгэж байна..."):
                try:
                    import base64
                    import json as _json
                    from src.utils.cloud_training import generate_colab_notebook
                    
                    target_col = st.session_state.get('target_col', '')
                    
                    # CSV-г base64 болгох
                    csv_bytes = df.to_csv(index=False).encode('utf-8')
                    csv_b64 = base64.b64encode(csv_bytes).decode('utf-8')
                    
                    notebook = generate_colab_notebook(
                        data_csv_base64=csv_b64,
                        target_column=target_col,
                        model_type=model_type,
                        params=params,
                    )
                    
                    notebook_json = _json.dumps(notebook, ensure_ascii=False, indent=2)
                    
                    st.download_button(
                        label=f"{ICONS['download']} Colab Notebook Татах (.ipynb)",
                        data=notebook_json,
                        file_name="xai_shap_cloud_training.ipynb",
                        mime="application/json",
                        width='stretch'
                    )
                    
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>{ICONS['check']} Notebook үүсгэгдлээ!</strong><br/>
                        {ICONS['bullet']} Файлыг татаж <a href="https://colab.research.google.com" target="_blank">Google Colab</a>-д нээнэ үү<br/>
                        {ICONS['bullet']} Runtime → Change runtime type → <strong>T4 GPU</strong> сонгоно уу<br/>
                        {ICONS['bullet']} Бүх cell-ийг дараалан ажиллуулна уу<br/>
                        {ICONS['bullet']} Сургалт дууссаны дараа үр дүнг автоматаар татна
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"{ICONS['warning']} Notebook үүсгэхэд алдаа: {e}")
        else:
            st.warning(f"{ICONS['warning']} Эхлээд өгөгдөл ачаална уу.")
    
    if local_train:
        with st.spinner(f"{model_type} загвар сургаж байна... Түр хүлээнэ үү."):
            try:
                framework.train_model(model_type=model_type, **params)
                st.session_state['model_trained'] = True
                
                # Амжилттай мэдээллийг session state-д хадгалах
                st.session_state['model_success_message'] = {
                    'model_type': model_type,
                    'model_name': type(framework.model).__name__
                }
                
                # Toast мэдэгдэл
                st.toast(f"{ICONS['check']} Загвар амжилттай сургагдлаа!", icon="✅")
                
                # UI шинэчлэх
                st.rerun()
                
            except Exception as e:
                st.error(f"{ICONS['warning']} Загвар сургахад алдаа: {e}")
                logger.error(f"Model training error: {e}")
    
    # Амжилттай мэдээллийг харуулах (rerun-ий дараа)
    if 'model_success_message' in st.session_state:
        msg = st.session_state['model_success_message']
        st.success(f"{ICONS['check']} {msg['model_name']} загвар амжилттай сургагдлаа!")
        del st.session_state['model_success_message']
    
    # Загварын үнэлгээ (хэрэв сургагдсан бол)
    model_trained = st.session_state.get('model_trained', False)
    
    if model_trained and framework.model is not None:
        st.subheader("Загварын Гүйцэтгэл")
        
        y_pred = framework.model.predict(framework.X_test)
        
        # Даалгаврын төрлийг тодорхойлох
        is_classification = False
        try:
            unique_values = np.unique(framework.y_test)
            if len(unique_values) <= 20:
                if np.all(unique_values == unique_values.astype(int)):
                    is_classification = True
        except (ValueError, TypeError):
            pass
        
        col1, col2, col3 = st.columns(3)
        
        if is_classification:
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            y_pred_rounded = y_pred.round()
            accuracy = accuracy_score(framework.y_test, y_pred_rounded)
            f1 = f1_score(framework.y_test, y_pred_rounded, average='weighted', zero_division=0)
            precision = precision_score(framework.y_test, y_pred_rounded, average='weighted', zero_division=0)
            recall = recall_score(framework.y_test, y_pred_rounded, average='weighted', zero_division=0)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Нарийвчлал (Accuracy)", f"{accuracy:.2%}")
            with col2:
                st.metric("F1 Оноо", f"{f1:.4f}")
            with col3:
                st.metric("Precision", f"{precision:.4f}")
            with col4:
                st.metric("Recall", f"{recall:.4f}")
        else:
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            r2 = r2_score(framework.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(framework.y_test, y_pred))
            mae = mean_absolute_error(framework.y_test, y_pred)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Оноо", f"{r2:.4f}")
            with col2:
                st.metric("RMSE", f"{rmse:.4f}")
            with col3:
                st.metric("MAE", f"{mae:.4f}")
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric("Загварын Төрөл", type(framework.model).__name__)
        with col_info2:
            st.metric("Шинж Чанарууд", len(framework.feature_names))


def render_explanation_section():
    """SHAP тайлбарын хэсгийг харуулах."""
    import streamlit as st
    
    st.markdown(f"""
    <div class="section-header">{ICONS['explain']} SHAP Тайлбарууд</div>
    """, unsafe_allow_html=True)
    
    framework = st.session_state.get('framework')
    
    # Framework state-аас session state-г синхрончлох
    model_trained = st.session_state.get('model_trained', False)
    if framework is not None and framework.model is not None:
        if not model_trained:
            st.session_state['model_trained'] = True
            model_trained = True
    
    # SHAP state-г синхрончлох
    explanations_generated = st.session_state.get('explanations_generated', False)
    explanations = st.session_state.get('explanations', {})
    
    if framework is not None and framework.shap_values is not None:
        if not explanations_generated:
            st.session_state['explanations_generated'] = True
    # Framework-т shap_values байхгүй бол session_state-аас сэргээх
    elif framework is not None and framework.shap_values is None and 'shap_values' in explanations:
        framework.shap_values = explanations['shap_values']
        framework._explanations = explanations
        if not explanations_generated:
            st.session_state['explanations_generated'] = True
    
    if not model_trained:
        st.markdown(f"""
        <div class="warning-box">
            {ICONS['warning']} <strong>Загвар шаардлагатай</strong><br/>
            Эхлээд "Загвар" хэсэгт очиж загвар сургаарай.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # SHAP тайлбар
    st.markdown(f"""
    <div class="info-box">
        <strong>SHAP (SHapley Additive exPlanations)</strong> нь тоглоомын онолд суурилсан загварын таамаглалыг тайлбарлах арга юм.<br/><br/>
        {ICONS['bullet']} <strong>Глобал тайлбар</strong> — бүх өгөгдөл дээр шинж чанар тус бүрийн нөлөөлөл хэр хэмжээ байгааг ерөнхийд нь харуулна<br/>
        {ICONS['bullet']} <strong>Локал тайлбар</strong> — нэг тодорхой дээжийн таамаглалд ямар шинж чанар яагаад нөлөөлснийг тайлбарлана<br/>
        {ICONS['bullet']} <strong>Хоёр хослуулан</strong> — дээрх хоёрыг нэгэн зэрэг үүсгэнэ
    </div>
    """, unsafe_allow_html=True)
    
    # Тохиргоо
    st.subheader("Тохиргоо")
    
    col1, col2 = st.columns(2)
    
    with col1:
        explanation_type = st.selectbox(
            "Тайлбарын Төрөл",
            ["both", "global", "local"],
            format_func=lambda x: {
                'both': 'Хоёр хослуулан (Global + Local)',
                'global': 'Глобал (Ерөнхий тайлбар)',
                'local': 'Локал (Дээж тус бүрийн)'
            }.get(x, x),
            help="Global: Бүх өгөгдолд ямар шинж чанар хамгийн их нөлөөтэй вэ? | Local: Нэг тодорхой дээжид яагаад ийм таамаглал өгсөн вэ? | Both: Хоёулангийг зэрэг үүсгэх"
        )
    
    with col2:
        max_features = st.slider("Харуулах Шинж Чанар", 5, 30, 20,
                                 help="Хамгийн их харуулах шинж чанарын тоо")
    
    # Тайлбарууд үүсгэх
    if st.button(f"{ICONS['play']} Тайлбарууд Үүсгэх", type="primary", width='stretch'):
        with st.spinner("SHAP утгуудыг тооцоолж байна... Энэ хэдэн минут үргэлжлэх болно."):
            try:
                explanations = framework.explain(explanation_type=explanation_type)
                st.session_state['explanations'] = explanations
                st.session_state['explanations_generated'] = True
                
                # Амжилттай мэдээллийг session state-д хадгалах
                st.session_state['shap_success_message'] = True
                
                # Toast мэдэгдэл
                st.toast(f"{ICONS['check']} SHAP тайлбарууд үүсгэгдлээ!", icon="✅")
                st.rerun()
                
            except Exception as e:
                st.error(f"{ICONS['warning']} Тайлбар үүсгэхэд алдаа: {e}")
                logger.error(f"Explanation error: {e}")
    
    # Амжилттай мэдээллийг харуулах (rerun-ий дараа)
    if 'shap_success_message' in st.session_state:
        st.success(f"{ICONS['check']} SHAP тайлбарууд амжилттай үүсгэгдлээ!")
        del st.session_state['shap_success_message']
    
    # Тайлбарууд харуулах
    explanations_generated = st.session_state.get('explanations_generated', False)
    explanations = st.session_state.get('explanations', {})
    
    # framework.shap_values эсвэл session_state дахь explanations-аас шалгах
    has_shap_values = (framework.shap_values is not None) or ('shap_values' in explanations)
    
    if explanations_generated and has_shap_values:
        # Глобал тайлбарууд
        if 'global' in explanations:
            st.subheader(f"{ICONS['chart']} Глобал Шинж Чанарын Ач Холбогдол")
            st.markdown(f"""
            <div class="info-box">
                Энэ хүснэгт шинж чанар бүр загварын таамаглалд <strong>ямар хэмжээгээр нөлөөлж байгаа</strong>г харуулна.
                Дээд эрэмбэлэгдсэн шинж чанарууд загварын үр дүнд хамгийн их нөлөөтэй.
            </div>
            """, unsafe_allow_html=True)
            importance_data = explanations['global']['feature_importance']
            df_importance = pd.DataFrame(importance_data)
            
            st.dataframe(
                df_importance.head(max_features),
                width='stretch'
            )
        
        # Локал тайлбарууд
        if 'local' in explanations:
            st.subheader(f"{ICONS['target']} Локал Тайлбарууд (Дээж Тус Бүрийн)")
            st.markdown(f"""
            <div class="info-box">
                Энд тус тусын дээжийн таамаглалд ямар шинж чанар <strong>ээрэг</strong> (таамаглалыг нэмэгдүүлсэн),
                ямар нь <strong>сөрөг</strong> (таамаглалыг бууруулсан) нөлөөтэй болохыг харуулна.
            </div>
            """, unsafe_allow_html=True)
            
            sample_idx = st.slider(
                "Дээж Сонгох",
                0, len(framework.X_test) - 1, 0,
                help="Тайлбар харах дээжийн дугаарыг сонгоно уу. Дээж бүр нь тест өгөгдлийн нэг мөр юм."
            )
            
            local_exp = explanations['local']['explanations']
            if sample_idx < len(local_exp):
                exp = local_exp[sample_idx]
                
                st.markdown(f"**Дээж #{exp['sample_index']}**")
                st.markdown(f"Суурь Утга (E[f(x)]): `{exp['base_value']:.4f}` — загварын дундаж гаралтын утга")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**{ICONS['up']} Эерэг Нөлөөтэй Шинж Чанарууд:**")
                    for item in exp['top_positive'][:5]:
                        shap_val = item.get('shap_value', item.get('contribution', 0))
                        st.markdown(f"<span class='feature-positive'>{ICONS['bullet']} {item['feature']}: +{shap_val:.4f}</span>", 
                                    unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"**{ICONS['down']} Сөрөг Нөлөөтэй Шинж Чанарууд:**")
                    for item in exp['top_negative'][:5]:
                        shap_val = item.get('shap_value', item.get('contribution', 0))
                        st.markdown(f"<span class='feature-negative'>{ICONS['bullet']} {item['feature']}: {shap_val:.4f}</span>",
                                    unsafe_allow_html=True)
        
        # Харгалзах тайлбар байхгүй бол
        if 'global' not in explanations and 'local' not in explanations:
            st.info(f"{ICONS['info']} Тайлбар үүсгэгдсэн боловч 'global' эсвэл 'local' өгөгдөл олдсонгүй.")
    elif explanations_generated and not has_shap_values:
        st.warning(f"{ICONS['warning']} Тайлбарууд үүсгэгдсэн боловч SHAP утгууд олдсонгүй. Дахин үүсгэнэ үү.")


def render_visualization_section():
    """Визуализацийн хэсгийг харуулах."""
    import streamlit as st
    
    st.markdown(f"""
    <div class="section-header">{ICONS['chart']} Визуализациуд</div>
    """, unsafe_allow_html=True)
    
    framework = st.session_state.get('framework')
    
    # Framework state-аас session state-г синхрончлох
    explanations_generated = st.session_state.get('explanations_generated', False)
    explanations = st.session_state.get('explanations', {})
    
    if framework is not None and framework.shap_values is not None:
        if not explanations_generated:
            st.session_state['explanations_generated'] = True
            explanations_generated = True
    # Framework-т shap_values байхгүй бол session_state-аас сэргээх
    elif framework is not None and framework.shap_values is None and 'shap_values' in explanations:
        framework.shap_values = explanations['shap_values']
        framework._explanations = explanations
        if not explanations_generated:
            st.session_state['explanations_generated'] = True
            explanations_generated = True
    
    # framework.shap_values эсвэл session_state дахь explanations-аас шалгах
    has_shap_values = (framework is not None and framework.shap_values is not None) or ('shap_values' in explanations)
    
    if not explanations_generated or not has_shap_values:
        st.markdown(f"""
        <div class="warning-box">
            {ICONS['warning']} <strong>SHAP тайлбар шаардлагатай</strong><br/>
            Эхлээд "Тайлбарууд" хэсэгт очиж SHAP тайлбар үүсгэнэ үү.
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown(f"""
    <div class="info-box">
        SHAP утгуудыг янз бүрийн графикаар дүрслэн харах боломжтой. График бүр загварын шийдвэрийг 
        <strong>өөр өнцгөөс</strong> тайлбарлана. Графикийн доор тайлбар гарч ирнэ.
    </div>
    """, unsafe_allow_html=True)
    
    # График төрөл сонгох
    col1, col2 = st.columns([1, 2])
    
    with col1:
        plot_type = st.selectbox(
            "Визуализаци Сонгох",
            ["summary", "bar", "waterfall", "heatmap", "violin", "dependence"],
            help="Үүсгэх графикийн төрлийг сонгоно уу"
        )
    
    with col2:
        plot_descriptions = {
            "summary": f"{ICONS['bullet']} <strong>Beeswarm</strong> — Шинж чанар бүрийн SHAP утгын тархалтыг цэг тус бүрээр харуулна. Улаан = их утга, цэнхэр = бага утга",
            "bar": f"{ICONS['bullet']} <strong>Bar (Баганан)</strong> — Шинж чанаруудыг дундаж нөлөөгөөр эрэмбэлсэн. Аль нь хамгийн чухал болохыг шууд харуулна",
            "waterfall": f"{ICONS['bullet']} <strong>Waterfall</strong> — Нэг дээжийн таамаглалыг алхам алхмаар задалж, шинж чанар тус бүр хэрхэн нэмсэн/хассаныг харуулна",
            "heatmap": f"{ICONS['bullet']} <strong>Heatmap</strong> — Бүх дээж × шинж чанаруудын SHAP утгын матриц. Өнгөний далайцаар нөлөөг харна",
            "violin": f"{ICONS['bullet']} <strong>Violin</strong> — SHAP утгын тархалтын хэлбэрийг бүлэг тус бүрээр харуулна. Тэгш хэмтэй эсэхийг шалгана",
            "dependence": f"{ICONS['bullet']} <strong>Dependence</strong> — Нэг шинж чанарын утга өөрчлөгдөхөд таамаглал хэрхэн өөрчлөгдөхийг харуулна"
        }
        st.markdown(f"""
        <div class="info-box">
            {plot_descriptions.get(plot_type, "")}
        </div>
        """, unsafe_allow_html=True)
    
    # Нэмэлт сонголтууд
    with st.expander(f"{ICONS['settings']} Нарийвчилсан Сонголтууд", expanded=False):
        max_display = st.slider("Шинж Чанарын Тоо", 5, 30, 15,
                                help="Графикт харуулах шинж чанарын хамгийн их тоо")
        
        sample_idx = 0
        feature = None
        
        if plot_type == "waterfall":
            sample_idx = st.slider(
                "Дээжийн Индекс",
                0, len(framework.X_test) - 1, 0,
                help="Waterfall график үүсгэх дээжийг сонгоно уу"
            )
        
        if plot_type == "dependence":
            feature = st.selectbox(
                "Шинж Чанар",
                framework.feature_names,
                help="Dependence график үүсгэх шинж чанарыг сонгоно уу"
            )
    
    # График үүсгэх
    if st.button(f"{ICONS['play']} График Үүсгэх", type="primary", width='stretch'):
        with st.spinner("Визуализаци үүсгэж байна..."):
            try:
                kwargs = {'max_display': max_display}
                
                if plot_type == "waterfall":
                    kwargs['sample_idx'] = sample_idx
                    # shap_values-г framework эсвэл session_state-аас авах
                    shap_vals = framework.shap_values if framework.shap_values is not None else explanations.get('shap_values')
                    if shap_vals is not None:
                        kwargs['base_value'] = float(np.mean(shap_vals.sum(axis=1)))
                
                if plot_type == "dependence" and feature:
                    kwargs['feature'] = feature
                
                fig = framework.visualize(plot_type=plot_type, **kwargs)
                st.plotly_chart(fig, use_container_width=True)
                
                # Графикийн тайлбар
                plot_explanations = {
                    "summary": f"""
                    <div class="info-box">
                        <strong>{ICONS['info']} Summary Plot (Beeswarm) Тайлбар:</strong><br/>
                        {ICONS['bullet']} Цэг бүр нэг дээж дэх нэг шинж чанарын SHAP утгыг илэрхийлнэ.<br/>
                        {ICONS['bullet']} <strong>Хэвтээ тэнхлэг (X):</strong> SHAP утга — эерэг утга таамаглалыг нэмэгдүүлж, сөрөг утга бууруулна.<br/>
                        {ICONS['bullet']} <strong>Босоо тэнхлэг (Y):</strong> Шинж чанарууд ач холбогдлоор эрэмблэгдсэн.<br/>
                        {ICONS['bullet']} <strong>Өнгө:</strong> Улаан = шинж чанарын утга өндөр, Цэнхэр = шинж чанарын утга бага.<br/>
                        {ICONS['bullet']} Өргөн тархалттай шинж чанарууд загварт илүү их нөлөөтэй.
                    </div>
                    """,
                    "bar": f"""
                    <div class="info-box">
                        <strong>{ICONS['info']} Bar Plot Тайлбар:</strong><br/>
                        {ICONS['bullet']} Шинж чанар бүрийн <strong>дундаж абсолют SHAP утга</strong>-ыг харуулна.<br/>
                        {ICONS['bullet']} Дээд шинж чанарууд загварын таамаглалд хамгийн их нөлөөлдөг.<br/>
                        {ICONS['bullet']} Энэ нь <strong>глобал шинж чанарын ач холбогдол</strong>-ыг илэрхийлнэ — бүх дээжүүдийн дундажаар.
                    </div>
                    """,
                    "waterfall": f"""
                    <div class="info-box">
                        <strong>{ICONS['info']} Waterfall Plot Тайлбар:</strong><br/>
                        {ICONS['bullet']} <strong>Нэг тодорхой дээж</strong>ийн таамаглалыг задлан харуулна.<br/>
                        {ICONS['bullet']} Доод талд суурь утга (E[f(x)]) — загварын дундаж гаралт.<br/>
                        {ICONS['bullet']} Улаан баганууд таамаглалыг <strong>нэмэгдүүлж</strong>, цэнхэр баганууд <strong>бууруулж</strong> байна.<br/>
                        {ICONS['bullet']} Дээд талд эцсийн таамаглалын утга f(x) харагдана.
                    </div>
                    """,
                    "heatmap": f"""
                    <div class="info-box">
                        <strong>{ICONS['info']} Heatmap Тайлбар:</strong><br/>
                        {ICONS['bullet']} Бүх дээжүүдийн шинж чанаруудын SHAP утгыг <strong>матриц</strong> хэлбэрээр харуулна.<br/>
                        {ICONS['bullet']} <strong>Мөр бүр:</strong> нэг дээж, <strong>Багана бүр:</strong> нэг шинж чанар.<br/>
                        {ICONS['bullet']} Өнгөний эрчим SHAP утгын хэмжээг илэрхийлнэ.<br/>
                        {ICONS['bullet']} Хэв маяг, бүлэглэл олоход тустай.
                    </div>
                    """,
                    "violin": f"""
                    <div class="info-box">
                        <strong>{ICONS['info']} Violin Plot Тайлбар:</strong><br/>
                        {ICONS['bullet']} Шинж чанар бүрийн SHAP утгын <strong>хуваарилалтын хэлбэр</strong>ийг харуулна.<br/>
                        {ICONS['bullet']} Өргөн хэсэг нь тухайн SHAP утгын давтамж ихтэйг илэрхийлнэ.<br/>
                        {ICONS['bullet']} Summary plot-тай төстэй боловч хуваарилалтын нарийвчилсан хэлбэрийг харуулна.
                    </div>
                    """,
                    "dependence": f"""
                    <div class="info-box">
                        <strong>{ICONS['info']} Dependence Plot Тайлбар:</strong><br/>
                        {ICONS['bullet']} Сонгосон шинж чанарын <strong>утга</strong> (X тэнхлэг) болон түүний <strong>SHAP утга</strong> (Y тэнхлэг)-ын хамаарлыг харуулна.<br/>
                        {ICONS['bullet']} Шинж чанарын утга өсөхөд SHAP утга хэрхэн өөрчлөгдөж байгааг ажиглах боломжтой.<br/>
                        {ICONS['bullet']} Шугаман бус хамаарлууд, босго утгууд зэргийг олоход тустай.
                    </div>
                    """
                }
                
                explanation_html = plot_explanations.get(plot_type, "")
                if explanation_html:
                    st.markdown(explanation_html, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"{ICONS['warning']} Визуализаци үүсгэхэд алдаа: {e}")
                logger.error(f"Visualization error: {e}")


def render_fairness_section():
    """Шударга байдлын шинжилгээний хэсгийг харуулах."""
    import streamlit as st
    
    st.markdown(f"""
    <div class="section-header">{ICONS['fairness']} Шударга Байдлын Шинжилгээ</div>
    """, unsafe_allow_html=True)
    
    framework = st.session_state.get('framework')
    
    # Framework state-аас session state-г синхрончлох
    model_trained = st.session_state.get('model_trained', False)
    if framework is not None and framework.model is not None:
        if not model_trained:
            st.session_state['model_trained'] = True
            model_trained = True
    
    if not model_trained:
        st.markdown(f"""
        <div class="warning-box">
            {ICONS['warning']} <strong>Загвар шаардлагатай</strong><br/>
            Эхлээд загвар сургаарай.
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown(f"""
    <div class="info-box">
        <strong>Шударга байдлын шинжилгээ</strong> нь таны AI загвар өөр өөр бүлгүүдэд тэгш хандаж байгааг шалгана.<br/><br/>
        {ICONS['bullet']} <strong>Зорилго:</strong> Загвар хүйс, нас, үндэстэн байдлаар ялгаварлахгүй байгааг шалгах<br/>
        {ICONS['bullet']} <strong>Арга:</strong> "Хамгаалагдсан атрибутууд" (жнь: хүйс, нас) сонгож, бүлэг тус бүр дээр хэмжигдэхүүн үдийг харьцуулна<br/>
        {ICONS['bullet']} Энэ нь <strong>Хариуцлагатай AI (Responsible AI)</strong>-ийн чухал бүрэлдэхүүн хэсэг юм
    </div>
    """, unsafe_allow_html=True)
    
    # Хамгаалагдсан атрибутууд
    st.subheader("Тохиргоо")
    
    if hasattr(framework, '_protected_attributes') and framework._protected_attributes:
        st.markdown(f"""
        <div class="success-box">
            <strong>Хамгаалагдсан атрибутууд:</strong> {', '.join(framework._protected_attributes)}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="warning-box">
            {ICONS['warning']} Хамгаалагдсан атрибут тохируулаагүй байна. 
            Өгөгдөл ачаалах үед хамгаалагдсан атрибутуудыг сонгоно уу.
        </div>
        """, unsafe_allow_html=True)
    
    # Шударга байдлын хэмжигдэхүүнүүдийн тайлбар
    st.subheader("Шударга Байдлын Хэмжигдэхүүнүүд")
    
    with st.expander(f"{ICONS['info']} Хэмжигдэхүүнүүдийн тайлбар", expanded=False):
        st.markdown(f"""
        **Demographic Parity (Хүн ам зүйн Тэнцвэрт байдал)**
        
        {ICONS['bullet']} Загвар өөр өөр бүлгүүдэд (жишээ нь: эр, эм) эерэг таамаглал өгөх хувь ижил байх ёстой
        
        {ICONS['bullet']} Жишээ нь: эрэгтэй 80% зээл авсан, эмэгтэй 50% зээл авсан бол шударга бус
        
        **Disparate Impact (80%-ийн дүрэм)**
        
        {ICONS['bullet']} Хамгийн бага хувьтай бүлгийн эерэг хувь нь хамгийн их хувьтай бүлгийн 80%-аас доош байвал алагчлал байна
        
        {ICONS['bullet']} Харьцаа 0.8-1.0 хооронд байвал шударга гэж үзнэ
        
        **Equalized Odds (Тэнцүүлсэн Магадлал)**
        
        {ICONS['bullet']} Бүлэг тус бүр дээр зөв таамагласан хувь (TPR) болон буруу таамагласан хувь (FPR) ижил байх ёстой
        
        {ICONS['bullet']} Жишээ нь: өвчин байгаа хүмүүсийг зөв олох хувь бүлэг бүрт ижил байх
        """)
    
    # Шударга байдлын үнэлгээг ажиллуулах
    if st.button(f"{ICONS['play']} Шинжилгээ Эхлүүлэх", type="primary", width='stretch'):
        with st.spinner("Шударга байдлыг шинжилж байна..."):
            try:
                results = framework.evaluate(include_fairness=True)
                
                if 'fairness' in results:
                    fairness_results = results['fairness']
                    st.session_state['fairness_results'] = fairness_results
                    st.success(f"{ICONS['check']} Шударга байдлын шинжилгээ дууслаа!")
                    st.rerun()
                else:
                    st.info(f"{ICONS['info']} Шударга байдлын хэмжигдэхүүн байхгүй. Хамгаалагдсан атрибутууд тохируулагдсан эсэхийг шалгана уу.")
                    
            except Exception as e:
                st.error(f"{ICONS['warning']} Шударга байдлын шинжилгээнд алдаа: {e}")
                logger.error(f"Fairness analysis error: {e}")
    
    # Үр дүнг харуулах
    if 'fairness_results' in st.session_state:
        fairness_results = st.session_state['fairness_results']
        
        # Анхааруулга байвал
        if 'warning' in fairness_results:
            st.markdown(f"""
            <div class="warning-box">
                {ICONS['warning']} <strong>Анхааруулга:</strong> {fairness_results['warning']}<br/>
                Өгөгдөл ачаалах хэсэгт буцаж хамгаалагдсан атрибутуудыг сонгоно уу.
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Ерөнхий Эерэг Хувь", f"{fairness_results.get('overall_positive_rate', 0):.2%}")
        
        # Бүрэн шинжилгээний үр дүн
        elif 'metrics_by_attribute' in fairness_results:
            # Ерөнхий шударга байдлын статус
            is_fair = fairness_results.get('overall_fairness', False)
            if is_fair:
                st.markdown(f"""
                <div class="success-box">
                    {ICONS['check']} <strong>Загвар шударга!</strong> Бүх хамгаалагдсан атрибутуудад шударга байдлын босго хангагдсан.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                    {ICONS['warning']} <strong>Шударга бус байдал илэрлээ!</strong> Зарим бүлгүүдэд тэгш бус хандлага байна.
                </div>
                """, unsafe_allow_html=True)
            
            # Атрибут бүрийн дэлгэрэнгүй
            for attr, metrics in fairness_results['metrics_by_attribute'].items():
                st.subheader(f"{ICONS['fairness']} {attr}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    dp_ratio = metrics.get('demographic_parity_ratio', 0)
                    st.metric(
                        "Demographic Parity Харьцаа",
                        f"{dp_ratio:.3f}",
                        help="1.0-д ойр байх тусам шударга"
                    )
                with col2:
                    di = metrics.get('disparate_impact', 0)
                    st.metric(
                        "Disparate Impact",
                        f"{di:.3f}",
                        help="0.8-аас дээш бол 80%-ийн дүрэм хангагдсан"
                    )
                with col3:
                    fair_label = f"{ICONS['check']} Тийм" if metrics.get('is_fair', False) else f"{ICONS['warning']} Үгүй"
                    st.metric("Шударга Эсэх", fair_label)
                
                # Бүлгийн тоон мэдээлэл
                group_metrics = metrics.get('group_metrics', {})
                if group_metrics:
                    with st.expander(f"{ICONS['info']} Бүлгийн Дэлгэрэнгүй ({attr})", expanded=True):
                        group_data = []
                        for group_name, gm in group_metrics.items():
                            group_data.append({
                                'Бүлэг': str(group_name),
                                'Хэмжээ': gm.get('size', 0),
                                'Эерэг Хувь': f"{gm.get('positive_rate', 0):.2%}",
                                'TPR': f"{gm['true_positive_rate']:.2%}" if gm.get('true_positive_rate') is not None else 'N/A',
                                'FPR': f"{gm['false_positive_rate']:.2%}" if gm.get('false_positive_rate') is not None else 'N/A'
                            })
                        st.dataframe(pd.DataFrame(group_data), width='stretch')
            
            # Зөвлөмжүүд
            recommendations = fairness_results.get('recommendations', [])
            if recommendations:
                st.subheader(f"{ICONS['info']} Зөвлөмжүүд")
                for rec in recommendations:
                    if rec.strip():
                        st.markdown(f"{ICONS['bullet']} {rec}")


# ============================================================================
# Helper Component Classes
# ============================================================================

class SidebarComponent:
    """Хажуугийн самбарын бүрэлдэхүүн класс."""
    
    @staticmethod
    def render():
        render_sidebar()


class MetricsComponent:
    """Хэмжигдэхүүн харуулах бүрэлдэхүүн."""
    
    @staticmethod
    def render(metrics: Dict[str, float]):
        import streamlit as st
        
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            with col:
                st.metric(name, f"{value:.4f}")


class ExplanationComponent:
    """Тайлбар харуулах бүрэлдэхүүн."""
    
    @staticmethod
    def render(explanation: Dict[str, Any]):
        import streamlit as st
        
        st.markdown(f"**Суурь Утга:** {explanation.get('base_value', 'N/A')}")
        
        contributions = explanation.get('contributions', [])
        for item in contributions[:10]:
            direction = ICONS['up'] if item['shap_value'] > 0 else ICONS['down']
            color_class = 'feature-positive' if item['shap_value'] > 0 else 'feature-negative'
            st.markdown(
                f"<span class='{color_class}'>{direction} <strong>{item['feature']}</strong>: {item['shap_value']:.4f}</span>",
                unsafe_allow_html=True
            )
