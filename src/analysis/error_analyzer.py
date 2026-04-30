"""
Error Analyzer - Алдааны шинжилгээ
===================================

Загварын алдаануудыг (False Positive, False Negative) 
дэлгэрэнгүй шинжилж, алдааны pattern олно.

Зохиогч: XAI-SHAP Framework
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import Counter
from src.utils.helpers import create_shap_explainer, extract_shap_values

logger = logging.getLogger(__name__)


@dataclass
class ErrorCase:
    """Алдааны кейсийн класс."""
    index: int
    true_label: int
    predicted_label: int
    confidence: float
    error_type: str  # 'FP', 'FN'
    features: np.ndarray
    shap_values: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'index': self.index,
            'true_label': int(self.true_label),
            'predicted_label': int(self.predicted_label),
            'confidence': float(self.confidence),
            'error_type': self.error_type
        }


@dataclass 
class ErrorPattern:
    """Алдааны pattern."""
    pattern_type: str
    description: str
    affected_count: int
    feature_conditions: Dict[str, Any]
    severity: str  # 'high', 'medium', 'low'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_type': self.pattern_type,
            'description': self.description,
            'affected_count': self.affected_count,
            'feature_conditions': self.feature_conditions,
            'severity': self.severity
        }


@dataclass
class ErrorAnalysisReport:
    """Алдааны шинжилгээний тайлан."""
    total_errors: int
    false_positives: int
    false_negatives: int
    fp_rate: float
    fn_rate: float
    error_cases: List[ErrorCase]
    patterns: List[ErrorPattern]
    feature_correlations: Dict[str, float]
    confusion_details: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Тайлангийн товчлол."""
        lines = ["=" * 60, "[SEARCH] АЛДААНЫ ШИНЖИЛГЭЭНИЙ ТАЙЛАН", "=" * 60, ""]
        
        # Ерөнхий статистик
        lines.append("[DATA] Ерөнхий статистик:")
        lines.append(f"  - Нийт алдаа: {self.total_errors}")
        lines.append(f"  - False Positive: {self.false_positives} ({self.fp_rate:.1%})")
        lines.append(f"  - False Negative: {self.false_negatives} ({self.fn_rate:.1%})")
        lines.append("")
        
        # Patterns
        if self.patterns:
            lines.append("[TARGET] Илэрсэн алдааны patterns:")
            for i, pattern in enumerate(self.patterns[:5], 1):
                severity_icon = {'high': '[HIGH]', 'medium': '[MED]', 'low': '[LOW]'}.get(pattern.severity, '[-]')
                lines.append(f"  {i}. {severity_icon} {pattern.description}")
                lines.append(f"     Нөлөөлсөн: {pattern.affected_count} жишээ")
        
        # Feature correlations
        if self.feature_correlations:
            lines.append("")
            lines.append("[CHART] Алдаатай хамааралтай features:")
            sorted_corr = sorted(self.feature_correlations.items(), key=lambda x: -abs(x[1]))
            for feature, corr in sorted_corr[:5]:
                direction = "+" if corr > 0 else "-"
                lines.append(f"  {direction} {feature}: {corr:.3f}")
        
        # Зөвлөмж
        if self.recommendations:
            lines.append("")
            lines.append("[TIP] Зөвлөмж:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")
        
        return "\n".join(lines)


class ErrorAnalyzer:
    """
    Загварын алдааг дэлгэрэнгүй шинжилнэ.
    
    Features:
    - False Positive / False Negative шинжилгээ
    - Алдааны pattern илрүүлэлт
    - Feature-error correlation
    - SHAP-based error explanation
    - Cluster-based error analysis
    
    Жишээ:
        >>> analyzer = ErrorAnalyzer(model, feature_names)
        >>> report = analyzer.analyze(X_test, y_test)
        >>> print(report.summary())
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        threshold: float = 0.5
    ):
        """
        ErrorAnalyzer эхлүүлэх.
        
        Параметрүүд:
            model: Сургасан загвар
            feature_names: Feature нэрс
            class_names: Ангийн нэрс
            threshold: Classification threshold
        """
        self.model = model
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.class_names = class_names or ['Negative', 'Positive']
        self.threshold = threshold
        
    def analyze(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        shap_values: Optional[np.ndarray] = None,
        calculate_shap: bool = True
    ) -> ErrorAnalysisReport:
        """
        Алдааны дэлгэрэнгүй шинжилгээ.
        
        Параметрүүд:
            X: Test features
            y_true: True labels
            shap_values: Pre-computed SHAP values
            calculate_shap: SHAP тооцох эсэх
            
        Буцаах:
            ErrorAnalysisReport
        """
        # Predictions
        y_pred, y_proba = self._get_predictions(X)
        
        # Error indices
        fp_indices, fn_indices = self._find_errors(y_true, y_pred)
        
        # SHAP values
        if calculate_shap and shap_values is None:
            shap_values = self._compute_shap(X)
        
        # Error cases
        error_cases = self._create_error_cases(
            X, y_true, y_pred, y_proba, 
            fp_indices, fn_indices, shap_values
        )
        
        # Patterns
        patterns = self._find_patterns(X, y_true, y_pred, fp_indices, fn_indices)
        
        # Feature correlations
        feature_correlations = self._compute_feature_correlations(X, y_true, y_pred)
        
        # Confusion details
        confusion_details = self._confusion_analysis(y_true, y_pred, y_proba)
        
        # Recommendations
        recommendations = self._generate_recommendations(
            len(fp_indices), len(fn_indices), patterns, feature_correlations
        )
        
        return ErrorAnalysisReport(
            total_errors=len(fp_indices) + len(fn_indices),
            false_positives=len(fp_indices),
            false_negatives=len(fn_indices),
            fp_rate=len(fp_indices) / max(sum(y_true == 0), 1),
            fn_rate=len(fn_indices) / max(sum(y_true == 1), 1),
            error_cases=error_cases,
            patterns=patterns,
            feature_correlations=feature_correlations,
            confusion_details=confusion_details,
            recommendations=recommendations
        )
    
    def _get_predictions(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Таамаглал авах."""
        if hasattr(self.model, 'predict_proba'):
            raw = self.model.predict_proba(X)
            if raw is not None:
                y_proba = np.atleast_1d(np.asarray(raw, dtype=float))
                if y_proba.ndim > 1:
                    y_proba = y_proba[:, 1]
                y_pred = (y_proba >= self.threshold).astype(int)
            else:
                y_pred = np.asarray(self.model.predict(X))
                y_proba = y_pred.astype(float)
        else:
            y_pred = np.asarray(self.model.predict(X))
            y_proba = y_pred.astype(float)
        
        return y_pred, y_proba
    
    def _find_errors(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Алдаануудыг олох."""
        fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
        fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
        return fp_indices, fn_indices
    
    def _compute_shap(self, X: np.ndarray) -> Optional[np.ndarray]:
        """SHAP values тооцох."""
        try:
            background = X[:min(100, len(X))]
            explainer = create_shap_explainer(self.model, background)
            return extract_shap_values(explainer, X[:min(200, len(X))])
        except Exception as e:
            logger.warning(f"SHAP тооцоход алдаа: {e}")
            return None
    
    def _create_error_cases(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        fp_indices: np.ndarray,
        fn_indices: np.ndarray,
        shap_values: Optional[np.ndarray]
    ) -> List[ErrorCase]:
        """Error cases үүсгэх."""
        cases = []
        
        for idx in fp_indices:
            cases.append(ErrorCase(
                index=int(idx),
                true_label=0,
                predicted_label=1,
                confidence=float(y_proba[idx]),
                error_type='FP',
                features=X[idx],
                shap_values=shap_values[idx] if shap_values is not None and idx < len(shap_values) else None
            ))
        
        for idx in fn_indices:
            cases.append(ErrorCase(
                index=int(idx),
                true_label=1,
                predicted_label=0,
                confidence=float(1 - y_proba[idx]),
                error_type='FN',
                features=X[idx],
                shap_values=shap_values[idx] if shap_values is not None and idx < len(shap_values) else None
            ))
        
        # Confidence-аар эрэмбэлэх (өндөр confidence алдаа = илүү санаа зовоосон)
        cases.sort(key=lambda x: -x.confidence)
        
        return cases
    
    def _find_patterns(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fp_indices: np.ndarray,
        fn_indices: np.ndarray
    ) -> List[ErrorPattern]:
        """Алдааны patterns олох."""
        patterns = []
        
        # FP patterns
        if len(fp_indices) >= 5:
            fp_patterns = self._analyze_error_group(X, fp_indices, 'FP')
            patterns.extend(fp_patterns)
        
        # FN patterns
        if len(fn_indices) >= 5:
            fn_patterns = self._analyze_error_group(X, fn_indices, 'FN')
            patterns.extend(fn_patterns)
        
        # High confidence errors
        _, y_proba = self._get_predictions(X)
        all_errors = np.concatenate([fp_indices, fn_indices])
        if len(all_errors) > 0:
            high_conf_errors = [i for i in all_errors if max(y_proba[i], 1 - y_proba[i]) > 0.8]
            if len(high_conf_errors) >= 3:
                patterns.append(ErrorPattern(
                    pattern_type='high_confidence_errors',
                    description=f'{len(high_conf_errors)} өндөр confidence-тэй (>80%) алдаа илэрлээ',
                    affected_count=len(high_conf_errors),
                    feature_conditions={'confidence': '>0.8'},
                    severity='high'
                ))
        
        # Boundary errors (close to threshold)
        boundary_errors = [i for i in all_errors if abs(y_proba[i] - self.threshold) < 0.1]
        if len(boundary_errors) >= 3:
            patterns.append(ErrorPattern(
                pattern_type='boundary_errors',
                description=f'{len(boundary_errors)} алдаа threshold орчимд (±0.1) байна',
                affected_count=len(boundary_errors),
                feature_conditions={'confidence': f'±0.1 from {self.threshold}'},
                severity='medium'
            ))
        
        # Sort by severity and count
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        patterns.sort(key=lambda x: (severity_order[x.severity], -x.affected_count))
        
        return patterns
    
    def _analyze_error_group(
        self,
        X: np.ndarray,
        error_indices: np.ndarray,
        error_type: str
    ) -> List[ErrorPattern]:
        """Алдааны бүлгийг шинжлэх."""
        patterns = []
        error_data = X[error_indices]
        all_data = X
        
        # Feature бүрээр шинжлэх
        for i, feature_name in enumerate(self.feature_names):
            error_values = error_data[:, i]
            all_values = all_data[:, i]
            
            error_mean = np.mean(error_values)
            error_std = np.std(error_values)
            all_mean = np.mean(all_values)
            all_std = np.std(all_values)
            
            # Z-score хэт их бол pattern
            if all_std > 0:
                z_score = abs(error_mean - all_mean) / all_std
                
                if z_score > 1.5:
                    direction = 'өндөр' if error_mean > all_mean else 'бага'
                    
                    patterns.append(ErrorPattern(
                        pattern_type=f'{error_type}_feature_bias',
                        description=f'{error_type} алдаанууд {feature_name} {direction} утгатай байх хандлагатай',
                        affected_count=len(error_indices),
                        feature_conditions={
                            'feature': feature_name,
                            'error_mean': float(error_mean),
                            'all_mean': float(all_mean),
                            'z_score': float(z_score)
                        },
                        severity='high' if z_score > 2.5 else 'medium'
                    ))
        
        # Extreme value errors
        for i, feature_name in enumerate(self.feature_names):
            error_values = error_data[:, i]
            q1, q3 = np.percentile(all_data[:, i], [25, 75])
            iqr = q3 - q1
            
            extreme_count = np.sum((error_values < q1 - 1.5 * iqr) | (error_values > q3 + 1.5 * iqr))
            if extreme_count >= len(error_indices) * 0.3:
                patterns.append(ErrorPattern(
                    pattern_type=f'{error_type}_extreme_values',
                    description=f'{error_type} алдааны {extreme_count} нь {feature_name}-д extreme утгатай',
                    affected_count=extreme_count,
                    feature_conditions={
                        'feature': feature_name,
                        'extreme_threshold': float(1.5 * iqr)
                    },
                    severity='medium'
                ))
        
        return patterns[:3]  # Top 3 patterns
    
    def _compute_feature_correlations(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Feature-алдаа correlation."""
        is_error = (y_true != y_pred).astype(float)
        
        correlations = {}
        for i, feature_name in enumerate(self.feature_names):
            # Point-biserial correlation
            feature_values = X[:, i]
            corr = np.corrcoef(feature_values, is_error)[0, 1]
            if not np.isnan(corr):
                correlations[feature_name] = float(corr)
        
        return correlations
    
    def _confusion_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Confusion matrix дэлгэрэнгүй шинжилгээ."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Confidence distribution by outcome
        correct_mask = y_true == y_pred
        
        return {
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'precision': tp / max(tp + fp, 1),
            'recall': tp / max(tp + fn, 1),
            'specificity': tn / max(tn + fp, 1),
            'avg_confidence_correct': float(np.mean([max(p, 1-p) for p, c in zip(y_proba, correct_mask) if c])) if any(correct_mask) else 0,
            'avg_confidence_error': float(np.mean([max(p, 1-p) for p, c in zip(y_proba, correct_mask) if not c])) if any(~correct_mask) else 0
        }
    
    def _generate_recommendations(
        self,
        n_fp: int,
        n_fn: int,
        patterns: List[ErrorPattern],
        correlations: Dict[str, float]
    ) -> List[str]:
        """Зөвлөмж үүсгэх."""
        recommendations = []
        
        # FP/FN balance
        if n_fp > n_fn * 2:
            recommendations.append(
                f"False Positive олон байна ({n_fp}). Threshold дээшлүүлэх эсвэл "
                "conservative prediction руу шилжүүлэхийг зөвлөе."
            )
        elif n_fn > n_fp * 2:
            recommendations.append(
                f"False Negative олон байна ({n_fn}). Threshold бууруулах эсвэл "
                "recall нэмэгдүүлэхэд анхаарна уу."
            )
        
        # High severity patterns
        high_severity = [p for p in patterns if p.severity == 'high']
        if high_severity:
            for pattern in high_severity[:2]:
                if 'feature' in pattern.feature_conditions:
                    feature = pattern.feature_conditions['feature']
                    recommendations.append(
                        f"'{feature}' feature-т анхаарна уу - алдаатай харилцаа илэрлээ."
                    )
        
        # High correlation features
        high_corr = [(f, c) for f, c in correlations.items() if abs(c) > 0.2]
        if high_corr:
            high_corr.sort(key=lambda x: -abs(x[1]))
            feature, corr = high_corr[0]
            recommendations.append(
                f"'{feature}' feature алдаатай хамааралтай (r={corr:.2f}). "
                "Feature engineering хийхийг зөвлөе."
            )
        
        # General recommendations
        if n_fp + n_fn > 0:
            total_errors = n_fp + n_fn
            if any(p.pattern_type == 'boundary_errors' for p in patterns):
                recommendations.append(
                    "Threshold орчимд олон алдаа байна. Model calibration хийхийг зөвлөе."
                )
            
            if any(p.pattern_type.endswith('extreme_values') for p in patterns):
                recommendations.append(
                    "Extreme утгууд дээр алдаа их байна. Outlier handling сайжруулна уу."
                )
        
        return recommendations
    
    def explain_error(
        self,
        error_case: ErrorCase,
        top_k: int = 5
    ) -> str:
        """
        Нэг алдааг дэлгэрэнгүй тайлбарлах.
        
        Параметрүүд:
            error_case: Тайлбарлах алдааны кейс
            top_k: Top features
            
        Буцаах:
            Тайлбарын текст
        """
        lines = [
            f"[SEARCH] Алдааны тайлбар (Index: {error_case.index})",
            "-" * 40,
            f"Төрөл: {'False Positive' if error_case.error_type == 'FP' else 'False Negative'}",
            f"Жинхэнэ утга: {self.class_names[error_case.true_label]}",
            f"Таамаглал: {self.class_names[error_case.predicted_label]}",
            f"Итгэл (Confidence): {error_case.confidence:.1%}",
            ""
        ]
        
        if error_case.shap_values is not None:
            lines.append("Алдаанд нөлөөлсөн features:")
            
            # Sort by absolute SHAP value
            feature_impacts = list(zip(self.feature_names, error_case.shap_values, error_case.features))
            feature_impacts.sort(key=lambda x: -abs(x[1]))
            
            for feature, shap_val, feature_val in feature_impacts[:top_k]:
                direction = "+" if shap_val > 0 else "-"
                influence = "нэмэгдүүлсэн" if shap_val > 0 else "бууруулсан"
                lines.append(
                    f"  {direction} {feature}: {feature_val:.2f} -> таамаглалыг {influence} ({shap_val:+.3f})"
                )
        
        return "\n".join(lines)
    
    def plot_error_analysis(self, report: ErrorAnalysisReport, output_path: Optional[str] = None):
        """
        Алдааны шинжилгээний график.
        
        Буцаах:
            HTML string эсвэл None
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Confusion Matrix',
                    'Error Distribution by Confidence',
                    'Feature-Error Correlations',
                    'Error Types'
                ),
                specs=[[{'type': 'heatmap'}, {'type': 'histogram'}],
                       [{'type': 'bar'}, {'type': 'pie'}]]
            )
            
            # Confusion Matrix
            cm = report.confusion_details['confusion_matrix']
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    x=self.class_names,
                    y=self.class_names,
                    colorscale='Blues',
                    text=cm,
                    texttemplate='%{text}',
                    textfont={'size': 16}
                ),
                row=1, col=1
            )
            
            # Confidence distribution
            fp_conf = [e.confidence for e in report.error_cases if e.error_type == 'FP']
            fn_conf = [e.confidence for e in report.error_cases if e.error_type == 'FN']
            
            if fp_conf:
                fig.add_trace(
                    go.Histogram(x=fp_conf, name='False Positive', 
                                opacity=0.7, marker_color='#ef4444'),
                    row=1, col=2
                )
            if fn_conf:
                fig.add_trace(
                    go.Histogram(x=fn_conf, name='False Negative',
                                opacity=0.7, marker_color='#3b82f6'),
                    row=1, col=2
                )
            
            # Feature correlations
            sorted_corr = sorted(report.feature_correlations.items(), key=lambda x: -abs(x[1]))[:10]
            features = [x[0] for x in sorted_corr]
            corrs = [x[1] for x in sorted_corr]
            colors = ['#ef4444' if c > 0 else '#3b82f6' for c in corrs]
            
            fig.add_trace(
                go.Bar(x=corrs, y=features, orientation='h',
                      marker_color=colors, name='Correlation'),
                row=2, col=1
            )
            
            # Error type pie
            fig.add_trace(
                go.Pie(
                    labels=['False Positive', 'False Negative', 'Correct'],
                    values=[
                        report.false_positives,
                        report.false_negatives,
                        len(report.error_cases) * 5  # Approximate correct
                    ],
                    marker_colors=['#ef4444', '#3b82f6', '#10b981']
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=700,
                title_text="Алдааны шинжилгээ",
                showlegend=True
            )
            
            if output_path:
                fig.write_html(output_path)
                return output_path
            else:
                return fig.to_html()
                
        except ImportError:
            logger.warning("Plotly суугаагүй байна")
            return None
