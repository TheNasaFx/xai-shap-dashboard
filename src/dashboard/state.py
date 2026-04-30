"""Dashboard session state helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional


STATUS_KEYS = (
    "data_loaded",
    "model_trained",
    "explanations_generated",
)

WORKFLOW_KEYS = (
    "dataset_name",
    "uploaded_data",
    "uploaded_file_signature",
    "target_col",
    "decision_threshold",
    "processing_info",
    "preprocessing_config",
    "explanations",
    "trained_models",
    "quality_report",
    "advanced_analysis_results",
    "fairness_results",
    "selected_model_name",
    "last_error",
)


def _default_state() -> Dict[str, Any]:
    return {
        "data_loaded": False,
        "model_trained": False,
        "explanations_generated": False,
        "dataset_name": None,
        "uploaded_data": None,
        "uploaded_file_signature": None,
        "target_col": None,
        "decision_threshold": 0.5,
        "processing_info": None,
        "preprocessing_config": {
            "missing_strategy": "median",
            "encoding_method": "onehot",
            "normalization_method": "standard",
            "test_size": 0.2,
        },
        "explanations": {},
        "trained_models": {},
        "quality_report": None,
        "advanced_analysis_results": {},
        "fairness_results": None,
        "selected_model_name": None,
        "last_error": None,
    }


def _new_framework():
    from src.core.framework import XAIFramework

    return XAIFramework()


def _has_explanation_payload(explanations: Dict[str, Any]) -> bool:
    return any(explanations.get(key) is not None for key in ("shap_values", "global", "local"))


def _clear_framework_explanations(framework) -> None:
    if framework is None:
        return

    framework.shap_values = None
    if hasattr(framework, "_explanations"):
        framework._explanations = {}
    if hasattr(framework, "explanations"):
        framework.explanations = {}


def ensure_dashboard_state(st_module, framework: Optional[Any] = None):
    """Ensure all expected session keys exist and derive readiness flags."""
    for key, value in _default_state().items():
        if key not in st_module.session_state:
            st_module.session_state[key] = deepcopy(value)

    if st_module.session_state.get("framework") is None:
        st_module.session_state["framework"] = framework if framework is not None else _new_framework()

    return sync_dashboard_state(st_module)


def sync_dashboard_state(st_module):
    """Synchronize readiness flags from the current framework payload."""
    framework = st_module.session_state.get("framework")
    explanations = st_module.session_state.get("explanations") or {}

    has_data = bool(
        framework is not None
        and framework.X_train is not None
        and framework.X_test is not None
        and framework.feature_names is not None
    )
    has_model = bool(has_data and framework.model is not None)
    has_shap = bool(
        has_model
        and (
            framework.shap_values is not None
            or _has_explanation_payload(explanations)
        )
    )

    st_module.session_state["data_loaded"] = has_data
    st_module.session_state["model_trained"] = has_model
    st_module.session_state["explanations_generated"] = has_shap

    return framework


def reset_dashboard_session(st_module, framework: Optional[Any] = None):
    """Clear the workflow state and start from a fresh framework instance."""
    for key in ("framework", *WORKFLOW_KEYS, *STATUS_KEYS):
        st_module.session_state.pop(key, None)

    return ensure_dashboard_state(st_module, framework=framework)


def replace_uploaded_data(
    st_module,
    dataframe,
    dataset_name: Optional[str] = None,
    uploaded_file_signature: Optional[str] = None,
):
    """Attach a newly uploaded raw dataset and drop every derived artifact."""
    reset_dashboard_session(st_module)
    st_module.session_state["dataset_name"] = dataset_name or "Custom dataset"
    st_module.session_state["uploaded_data"] = dataframe
    st_module.session_state["uploaded_file_signature"] = uploaded_file_signature
    return sync_dashboard_state(st_module)


def invalidate_after_data_load(st_module, target_col: Optional[str] = None):
    """Keep processed data but clear every artifact derived from an older model run."""
    framework = st_module.session_state.get("framework")
    _clear_framework_explanations(framework)

    if target_col is not None:
        st_module.session_state["target_col"] = target_col

    st_module.session_state["decision_threshold"] = 0.5
    st_module.session_state["processing_info"] = getattr(framework, "processing_info", None)

    st_module.session_state["trained_models"] = {}
    st_module.session_state["selected_model_name"] = None
    st_module.session_state["explanations"] = {}
    st_module.session_state["advanced_analysis_results"] = {}
    st_module.session_state["fairness_results"] = None

    return sync_dashboard_state(st_module)


def invalidate_after_model_change(st_module):
    """Clear SHAP and fairness artifacts after the active model changes."""
    framework = st_module.session_state.get("framework")
    _clear_framework_explanations(framework)

    st_module.session_state["decision_threshold"] = 0.5
    st_module.session_state["explanations"] = {}
    st_module.session_state["advanced_analysis_results"] = {}
    st_module.session_state["fairness_results"] = None

    return sync_dashboard_state(st_module)


def clear_model_state(st_module):
    """Remove the active model and every artifact that depends on it."""
    framework = st_module.session_state.get("framework")
    if framework is not None:
        framework.model = None

    _clear_framework_explanations(framework)

    st_module.session_state["decision_threshold"] = 0.5
    st_module.session_state["trained_models"] = {}
    st_module.session_state["selected_model_name"] = None
    st_module.session_state["explanations"] = {}
    st_module.session_state["advanced_analysis_results"] = {}
    st_module.session_state["fairness_results"] = None

    return sync_dashboard_state(st_module)


def get_workflow_status(st_module) -> Dict[str, Any]:
    """Return a compact snapshot used by the dashboard header and status UI."""
    framework = sync_dashboard_state(st_module)
    uploaded_data = st_module.session_state.get("uploaded_data")
    trained_models = st_module.session_state.get("trained_models") or {}

    data_loaded = bool(st_module.session_state.get("data_loaded"))
    model_trained = bool(st_module.session_state.get("model_trained"))
    explanations_generated = bool(st_module.session_state.get("explanations_generated"))

    return {
        "framework": framework,
        "dataset_name": st_module.session_state.get("dataset_name") or (
            "Custom dataset" if uploaded_data is not None else None
        ),
        "uploaded_data": uploaded_data,
        "row_count": int(len(uploaded_data)) if uploaded_data is not None else 0,
        "column_count": int(uploaded_data.shape[1]) if uploaded_data is not None else 0,
        "target_col": st_module.session_state.get("target_col"),
        "selected_model_name": st_module.session_state.get("selected_model_name"),
        "trained_model_count": len(trained_models),
        "data_loaded": data_loaded,
        "model_trained": model_trained,
        "explanations_generated": explanations_generated,
        "completed_steps": sum(int(flag) for flag in (data_loaded, model_trained, explanations_generated)),
    }