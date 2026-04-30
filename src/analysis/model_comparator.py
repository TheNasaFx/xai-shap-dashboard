"""
Model Comparator - Загварын харьцуулалт
========================================

Олон загварыг гүйцэтгэл, SHAP нийцтэй байдал, хугацаа зэргээр
харьцуулж, хамгийн сайн загварыг санал болгоно.

Зохиогч: XAI-SHAP Framework
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
from src.utils.helpers import create_shap_explainer, extract_shap_values, unwrap_model_for_shap

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Загварын үр дүнгийн класс."""
    name: str
    model: Any
    train_time: float
    inference_time: float
    metrics: Dict[str, float]
    cv_scores: Optional[np.ndarray] = None
    shap_values: Optional[np.ndarray] = None
    feature_importance: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary руу хөрвүүлэх."""
        return {
            'name': self.name,
            'train_time': self.train_time,
            'inference_time': self.inference_time,
            'metrics': self.metrics,
            'cv_mean': float(np.mean(self.cv_scores)) if self.cv_scores is not None else None,
            'cv_std': float(np.std(self.cv_scores)) if self.cv_scores is not None else None
        }


@dataclass
class ComparisonReport:
    """Харьцуулалтын тайлангийн класс."""
    results: List[ModelResult]
    best_model: str
    ranking: List[Tuple[str, float]]
    shap_agreement: Optional[Dict[str, float]] = None
    recommendations: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Тайлангийн товчлол."""
        lines = ["=" * 60, "[DATA] ЗАГВАРЫН ХАРЬЦУУЛАЛТЫН ТАЙЛАН", "=" * 60, ""]
        
        # Хамгийн сайн загвар
        lines.append(f"[BEST] Хамгийн сайн загвар: {self.best_model}")
        lines.append("")
        
        # Эрэмбэ
        lines.append("[RANK] Эрэмбэ:")
        for rank, (name, score) in enumerate(self.ranking, 1):
            lines.append(f"  {rank}. {name}: {score:.4f}")
        
        # Metrics хүснэгт
        lines.append("")
        lines.append("[TABLE] Гүйцэтгэлийн хүснэгт:")
        lines.append("-" * 60)
        
        headers = ["Загвар", "Accuracy", "F1", "AUC", "Train (s)", "Infer (ms)"]
        lines.append("  " + " | ".join(f"{h:>10}" for h in headers))
        lines.append("-" * 60)
        
        for result in self.results:
            row = [
                result.name[:10],
                f"{result.metrics.get('accuracy', 0):.4f}",
                f"{result.metrics.get('f1', 0):.4f}",
                f"{result.metrics.get('roc_auc', 0):.4f}",
                f"{result.train_time:.2f}",
                f"{result.inference_time * 1000:.2f}"
            ]
            lines.append("  " + " | ".join(f"{v:>10}" for v in row))
        
        # SHAP agreement
        if self.shap_agreement:
            lines.append("")
            lines.append("[SHAP] SHAP нийцтэй байдал:")
            for pair, agreement in self.shap_agreement.items():
                lines.append(f"  {pair}: {agreement:.2%}")
        
        # Зөвлөмж
        if self.recommendations:
            lines.append("")
            lines.append("[TIP] Зөвлөмж:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")
        
        return "\n".join(lines)


class ModelComparator:
    """
    Олон загварыг харьцуулагч.
    
    Features:
    - Multi-metric comparison (accuracy, F1, AUC, etc.)
    - Cross-validation
    - Training & inference time measurement
    - SHAP agreement analysis
    - Automatic model selection
    
    Жишээ:
        >>> comparator = ModelComparator()
        >>> comparator.add_model('XGBoost', xgb_model)
        >>> comparator.add_model('RandomForest', rf_model)
        >>> report = comparator.compare(X_train, X_test, y_train, y_test)
        >>> print(report.summary())
    """
    
    def __init__(
        self,
        task_type: str = 'classification',
        primary_metric: str = 'f1',
        cv_folds: int = 5,
        calculate_shap: bool = True
    ):
        """
        ModelComparator эхлүүлэх.
        
        Параметрүүд:
            task_type: 'classification' эсвэл 'regression'
            primary_metric: Эрэмбэлэхэд ашиглах гол метрик
            cv_folds: Cross-validation folds
            calculate_shap: SHAP тооцох эсэх
        """
        self.task_type = task_type
        self.primary_metric = primary_metric
        self.cv_folds = cv_folds
        self.calculate_shap = calculate_shap
        self.models: Dict[str, Any] = {}
        self.results: List[ModelResult] = []
        
    def add_model(
        self,
        name: str,
        model: Any,
        train_func: Optional[Callable] = None
    ):
        """
        Загвар нэмэх.
        
        Параметрүүд:
            name: Загварын нэр
            model: Загварын объект
            train_func: Custom training function (optional)
        """
        self.models[name] = {
            'model': model,
            'train_func': train_func
        }
        logger.info(f"Загвар нэмэгдлээ: {name}")
    
    def compare(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> ComparisonReport:
        """
        Бүх загварыг харьцуулах.
        
        Параметрүүд:
            X_train: Сургалтын өгөгдөл
            X_test: Тестийн өгөгдөл
            y_train: Сургалтын label
            y_test: Тестийн label
            feature_names: Feature нэрс
            
        Буцаах:
            ComparisonReport
        """
        self.results = []
        
        for name, model_info in self.models.items():
            logger.info(f"Харьцуулж байна: {name}")
            
            try:
                result = self._evaluate_model(
                    name,
                    model_info['model'],
                    model_info['train_func'],
                    X_train, X_test, y_train, y_test,
                    feature_names
                )
                self.results.append(result)
            except Exception as e:
                logger.error(f"{name} үнэлэхэд алдаа: {e}")
        
        # SHAP agreement тооцох
        shap_agreement = None
        if self.calculate_shap and len(self.results) > 1:
            shap_agreement = self._calculate_shap_agreement()
        
        # Ranking тооцох
        ranking = self._compute_ranking()
        best_model = ranking[0][0] if ranking else None
        
        # Зөвлөмж
        recommendations = self._generate_recommendations()
        
        return ComparisonReport(
            results=self.results,
            best_model=best_model,
            ranking=ranking,
            shap_agreement=shap_agreement,
            recommendations=recommendations
        )
    
    def _evaluate_model(
        self,
        name: str,
        model: Any,
        train_func: Optional[Callable],
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]]
    ) -> ModelResult:
        """Нэг загварыг үнэлэх."""
        # Training time
        start_time = time.time()
        if train_func:
            train_func(model, X_train, y_train)
        else:
            model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Inference time
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = (time.time() - start_time) / len(X_test)
        
        # Metrics
        if self.task_type == 'classification':
            metrics = self._classification_metrics(model, X_test, y_test, y_pred)
        else:
            metrics = self._regression_metrics(y_test, y_pred)
        
        # Cross-validation
        cv_scores = None
        try:
            scoring = 'f1' if self.task_type == 'classification' else 'r2'
            cv_scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring=scoring)
        except Exception as e:
            logger.warning(f"CV тооцоход алдаа: {e}")
        
        # SHAP values
        shap_values = None
        feature_importance = None
        if self.calculate_shap:
            try:
                actual_model, _ = unwrap_model_for_shap(model)
                if hasattr(actual_model, 'feature_importances_'):
                    feature_importance = actual_model.feature_importances_

                background = X_train[:min(100, len(X_train))]
                explainer = create_shap_explainer(model, background)
                shap_values = extract_shap_values(explainer, X_test[:min(100, len(X_test))])
            except Exception as e:
                logger.warning(f"SHAP тооцоход алдаа: {e}")
        
        return ModelResult(
            name=name,
            model=model,
            train_time=train_time,
            inference_time=inference_time,
            metrics=metrics,
            cv_scores=cv_scores,
            shap_values=shap_values,
            feature_importance=feature_importance
        )
    
    def _classification_metrics(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Classification метрикүүд."""
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
                if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
            except:
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    def _regression_metrics(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Regression метрикүүд."""
        return {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    
    def _calculate_shap_agreement(self) -> Dict[str, float]:
        """SHAP values-ийн нийцтэй байдлыг тооцох."""
        agreement = {}
        
        models_with_shap = [r for r in self.results if r.shap_values is not None]
        
        for i in range(len(models_with_shap)):
            for j in range(i + 1, len(models_with_shap)):
                model1 = models_with_shap[i]
                model2 = models_with_shap[j]
                
                shap1 = model1.shap_values
                shap2 = model2.shap_values
                
                # Хэмжээ тохируулах
                min_samples = min(len(shap1), len(shap2))
                shap1 = shap1[:min_samples]
                shap2 = shap2[:min_samples]
                
                # Feature importance ranking agreement
                importance1 = np.abs(shap1).mean(axis=0)
                importance2 = np.abs(shap2).mean(axis=0)
                
                rank1 = np.argsort(importance1)[::-1]
                rank2 = np.argsort(importance2)[::-1]
                
                # Kendall's tau
                from scipy.stats import kendalltau
                tau, _ = kendalltau(rank1, rank2)
                
                pair_name = f"{model1.name} vs {model2.name}"
                agreement[pair_name] = (tau + 1) / 2  # 0-1 хооронд
        
        return agreement
    
    def _compute_ranking(self) -> List[Tuple[str, float]]:
        """Загваруудыг эрэмбэлэх."""
        scores = []
        
        for result in self.results:
            metric_value = result.metrics.get(self.primary_metric, 0)
            
            # CV score-г оруулах
            cv_score = 0
            if result.cv_scores is not None:
                cv_score = np.mean(result.cv_scores)
            
            # Weighted score
            combined_score = metric_value * 0.6 + cv_score * 0.4
            scores.append((result.name, combined_score))
        
        scores.sort(key=lambda x: -x[1])
        return scores
    
    def _generate_recommendations(self) -> List[str]:
        """Зөвлөмж үүсгэх."""
        if not self.results:
            return []
        
        recommendations = []
        
        # Хамгийн сайн загвар
        best = max(self.results, key=lambda r: r.metrics.get(self.primary_metric, 0))
        recommendations.append(
            f"'{best.name}' хамгийн өндөр {self.primary_metric} утгатай байна."
        )
        
        # Хурд
        fastest = min(self.results, key=lambda r: r.train_time)
        if fastest.name != best.name:
            recommendations.append(
                f"'{fastest.name}' хамгийн хурдан сургагддаг ({fastest.train_time:.2f}s)."
            )
        
        # Inference speed
        fastest_infer = min(self.results, key=lambda r: r.inference_time)
        if fastest_infer.name != best.name:
            recommendations.append(
                f"'{fastest_infer.name}' хамгийн хурдан таамаглал хийдэг "
                f"({fastest_infer.inference_time * 1000:.2f}ms)."
            )
        
        # CV stability
        most_stable = None
        min_std = float('inf')
        for r in self.results:
            if r.cv_scores is not None:
                std = np.std(r.cv_scores)
                if std < min_std:
                    min_std = std
                    most_stable = r
        
        if most_stable and most_stable.name != best.name:
            recommendations.append(
                f"'{most_stable.name}' хамгийн тогтвортой CV үр дүнтэй "
                f"(std={min_std:.4f})."
            )
        
        # Trade-off зөвлөмж
        if len(self.results) >= 2:
            sorted_by_metric = sorted(
                self.results, 
                key=lambda r: -r.metrics.get(self.primary_metric, 0)
            )
            sorted_by_speed = sorted(self.results, key=lambda r: r.train_time)
            
            if sorted_by_metric[0].name != sorted_by_speed[0].name:
                recommendations.append(
                    f"Accuracy vs Speed trade-off: '{sorted_by_metric[0].name}' нарийвчлалаар, "
                    f"'{sorted_by_speed[0].name}' хурдаар илүү."
                )
        
        return recommendations
    
    def plot_comparison(self, output_path: Optional[str] = None) -> Optional[str]:
        """
        Харьцуулалтын график.
        
        Буцаах:
            HTML string (plotly) эсвэл None
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Гүйцэтгэлийн метрикүүд',
                    'Сургалтын хугацаа',
                    'CV Scores Distribution',
                    'Metric Radar'
                )
            )
            
            names = [r.name for r in self.results]
            colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
            
            # Bar chart - Metrics
            metrics_to_plot = ['accuracy', 'f1', 'precision', 'recall']
            if self.task_type == 'regression':
                metrics_to_plot = ['r2', 'mae', 'rmse']
            
            for i, metric in enumerate(metrics_to_plot):
                values = [r.metrics.get(metric, 0) for r in self.results]
                fig.add_trace(
                    go.Bar(name=metric, x=names, y=values, 
                           marker_color=colors[i % len(colors)]),
                    row=1, col=1
                )
            
            # Training time
            train_times = [r.train_time for r in self.results]
            fig.add_trace(
                go.Bar(x=names, y=train_times, marker_color='#10b981',
                       name='Train Time'),
                row=1, col=2
            )
            
            # CV Scores boxplot
            for i, result in enumerate(self.results):
                if result.cv_scores is not None:
                    fig.add_trace(
                        go.Box(y=result.cv_scores, name=result.name,
                               marker_color=colors[i % len(colors)]),
                        row=2, col=1
                    )
            
            # Radar chart
            if self.task_type == 'classification':
                categories = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
            else:
                categories = ['r2', 'mae', 'rmse']
            
            for i, result in enumerate(self.results):
                values = [result.metrics.get(c, 0) for c in categories]
                values.append(values[0])  # Close the polygon
                
                fig.add_trace(
                    go.Scatterpolar(
                        r=values,
                        theta=categories + [categories[0]],
                        name=result.name,
                        fill='toself',
                        opacity=0.5
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=800,
                title_text="Загварын харьцуулалт",
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
    
    def get_best_model(self) -> Tuple[str, Any]:
        """Хамгийн сайн загварыг буцаах."""
        if not self.results:
            raise ValueError("Харьцуулалт хийгдээгүй байна")
        
        best = max(self.results, key=lambda r: r.metrics.get(self.primary_metric, 0))
        return best.name, best.model
    
    def feature_importance_comparison(self) -> pd.DataFrame:
        """Feature importance харьцуулалт."""
        importance_data = {}
        
        for result in self.results:
            if result.shap_values is not None:
                importance = np.abs(result.shap_values).mean(axis=0)
                importance_data[result.name] = importance
            elif result.feature_importance is not None:
                importance_data[result.name] = result.feature_importance
        
        if not importance_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(importance_data)
        
        # Rank correlation
        if len(importance_data) > 1:
            rankings = df.rank(ascending=False)
            df['rank_variance'] = rankings.var(axis=1)
        
        return df
