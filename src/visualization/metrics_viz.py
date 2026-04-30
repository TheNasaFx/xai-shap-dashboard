"""
Model Metrics Visualizer - Загварын гүйцэтгэлийн визуализаци
=============================================================

ROC curves, PR curves, Confusion Matrix, Calibration curves,
Learning curves гэх мэт загварын үнэлгээний визуализаци.

Зохиогч: XAI-SHAP Framework
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MetricsVisualizer:
    """
    Загварын гүйцэтгэлийн визуализаци.
    
    Боломжит графикууд:
    - ROC Curve (AUC)
    - Precision-Recall Curve
    - Confusion Matrix Heatmap
    - Calibration Curve
    - Learning Curves
    - Feature Correlation Heatmap
    - Prediction Distribution
    - Threshold Analysis
    
    Жишээ:
        >>> visualizer = MetricsVisualizer()
        >>> fig = visualizer.plot_roc_curve(y_true, y_proba)
        >>> fig.show()
    """
    
    def __init__(self, theme: str = 'plotly_white'):
        """
        MetricsVisualizer эхлүүлэх.
        
        Параметрүүд:
            theme: Plotly template
        """
        self.theme = theme
        self.colors = {
            'primary': '#3b82f6',
            'secondary': '#10b981',
            'danger': '#ef4444',
            'warning': '#f59e0b',
            'info': '#6366f1',
            'success': '#22c55e'
        }
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "ROC Curve",
        show_threshold: bool = True,
        optimal_threshold: bool = True
    ):
        """
        ROC (Receiver Operating Characteristic) муруй зурах.
        
        Параметрүүд:
            y_true: Бодит label
            y_proba: Таамагласан магадлал
            title: Графикийн гарчиг
            show_threshold: Threshold-уудыг харуулах
            optimal_threshold: Оптимал threshold-г харуулах
            
        Буцаах:
            Plotly figure
        """
        import plotly.graph_objects as go
        from sklearn.metrics import roc_curve, roc_auc_score
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        
        fig = go.Figure()
        
        # ROC Curve
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_score:.4f})',
            line=dict(color=self.colors['primary'], width=2),
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<br>Threshold: %{customdata:.3f}',
            customdata=thresholds
        ))
        
        # Random classifier line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        # Optimal threshold (Youden's J statistic)
        if optimal_threshold:
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_thresh = thresholds[optimal_idx]
            
            fig.add_trace(go.Scatter(
                x=[fpr[optimal_idx]],
                y=[tpr[optimal_idx]],
                mode='markers',
                name=f'Optimal Threshold ({optimal_thresh:.3f})',
                marker=dict(color=self.colors['danger'], size=12, symbol='star')
            ))
        
        # AUC fill
        fig.add_trace(go.Scatter(
            x=list(fpr) + [1, 0],
            y=list(tpr) + [0, 0],
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.1)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=dict(
                text=f'{title}<br><sub>Area Under Curve (AUC) = {auc_score:.4f}</sub>',
                x=0.5
            ),
            xaxis_title='False Positive Rate (1 - Specificity)',
            yaxis_title='True Positive Rate (Sensitivity)',
            template=self.theme,
            height=500,
            width=600,
            legend=dict(x=0.6, y=0.1),
            xaxis=dict(range=[-0.02, 1.02]),
            yaxis=dict(range=[-0.02, 1.02])
        )
        
        return fig
    
    def plot_pr_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "Precision-Recall Curve"
    ):
        """
        Precision-Recall муруй зурах.
        
        Параметрүүд:
            y_true: Бодит label
            y_proba: Таамагласан магадлал
            title: Графикийн гарчиг
            
        Буцаах:
            Plotly figure
        """
        import plotly.graph_objects as go
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        ap_score = average_precision_score(y_true, y_proba)
        
        # Baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        
        fig = go.Figure()
        
        # PR Curve
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'PR Curve (AP = {ap_score:.4f})',
            line=dict(color=self.colors['secondary'], width=2),
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.1)'
        ))
        
        # Baseline
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[baseline, baseline],
            mode='lines',
            name=f'Baseline ({baseline:.3f})',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        # F1 iso-curves
        for f1_score in [0.2, 0.4, 0.6, 0.8]:
            x = np.linspace(0.01, 1, 100)
            y = f1_score * x / (2 * x - f1_score)
            y = np.where((y > 0) & (y <= 1), y, np.nan)
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name=f'F1={f1_score}',
                line=dict(color='lightgray', width=0.5, dash='dot'),
                showlegend=False
            ))
        
        fig.update_layout(
            title=dict(
                text=f'{title}<br><sub>Average Precision = {ap_score:.4f}</sub>',
                x=0.5
            ),
            xaxis_title='Recall (Sensitivity)',
            yaxis_title='Precision',
            template=self.theme,
            height=500,
            width=600,
            legend=dict(x=0.1, y=0.1),
            xaxis=dict(range=[-0.02, 1.02]),
            yaxis=dict(range=[-0.02, 1.02])
        )
        
        return fig
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        normalize: bool = True,
        title: str = "Confusion Matrix"
    ):
        """
        Confusion Matrix heatmap зурах.
        
        Параметрүүд:
            y_true: Бодит label
            y_pred: Таамагласан label
            class_names: Ангийн нэрс
            normalize: Normalize хийх эсэх
            title: Графикийн гарчиг
            
        Буцаах:
            Plotly figure
        """
        import plotly.graph_objects as go
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_display = cm_normalized
            fmt = '.2%'
        else:
            cm_display = cm
            fmt = 'd'
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        # Annotation text
        annotations = []
        for i, row in enumerate(cm_display):
            for j, val in enumerate(row):
                text = f'{val:.1%}' if normalize else f'{val}'
                annotations.append(dict(
                    x=class_names[j],
                    y=class_names[i],
                    text=text,
                    showarrow=False,
                    font=dict(
                        color='white' if cm_display[i, j] > cm_display.max() / 2 else 'black',
                        size=14
                    )
                ))
        
        fig = go.Figure(data=go.Heatmap(
            z=cm_display,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title='Rate' if normalize else 'Count')
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            template=self.theme,
            height=450,
            width=500,
            annotations=annotations
        )
        
        # Reverse y-axis so True label 0 is at top
        fig.update_yaxes(autorange='reversed')
        
        return fig
    
    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
        title: str = "Calibration Curve"
    ):
        """
        Calibration (Reliability) муруй зурах.
        
        Загварын магадлалын үнэн зөв байдлыг шалгана.
        
        Параметрүүд:
            y_true: Бодит label
            y_proba: Таамагласан магадлал
            n_bins: Bin-ийн тоо
            title: Графикийн гарчиг
            
        Буцаах:
            Plotly figure
        """
        import plotly.graph_objects as go
        from sklearn.calibration import calibration_curve
        
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
        
        # Brier score
        brier_score = np.mean((y_proba - y_true) ** 2)
        
        fig = go.Figure()
        
        # Perfect calibration
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfectly Calibrated',
            line=dict(color='gray', dash='dash')
        ))
        
        # Model calibration
        fig.add_trace(go.Scatter(
            x=prob_pred,
            y=prob_true,
            mode='lines+markers',
            name=f'Model (Brier={brier_score:.4f})',
            line=dict(color=self.colors['primary'], width=2),
            marker=dict(size=8)
        ))
        
        # Histogram of predictions
        fig.add_trace(go.Histogram(
            x=y_proba,
            name='Prediction Distribution',
            yaxis='y2',
            opacity=0.3,
            marker_color=self.colors['info']
        ))
        
        fig.update_layout(
            title=dict(
                text=f'{title}<br><sub>Brier Score = {brier_score:.4f} (lower is better)</sub>',
                x=0.5
            ),
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            template=self.theme,
            height=500,
            width=600,
            legend=dict(x=0.6, y=0.15),
            yaxis2=dict(
                title='Count',
                overlaying='y',
                side='right',
                showgrid=False
            )
        )
        
        return fig
    
    def plot_learning_curve(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        n_jobs: int = -1,
        train_sizes: np.ndarray = None,
        title: str = "Learning Curve"
    ):
        """
        Learning curve зурах.
        
        Сургалтын хэмжээ нэмэгдэхэд гүйцэтгэл хэрхэн өөрчлөгдөхийг харуулна.
        
        Параметрүүд:
            model: Загвар
            X: Features
            y: Labels
            cv: Cross-validation folds
            n_jobs: Parallel jobs
            train_sizes: Сургалтын хэмжээнүүд
            title: Графикийн гарчиг
            
        Буцаах:
            Plotly figure
        """
        import plotly.graph_objects as go
        from sklearn.model_selection import learning_curve
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=n_jobs,
            train_sizes=train_sizes, scoring='accuracy'
        )
        
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
        
        fig = go.Figure()
        
        # Training score
        fig.add_trace(go.Scatter(
            x=train_sizes_abs,
            y=train_mean,
            mode='lines+markers',
            name='Training Score',
            line=dict(color=self.colors['primary']),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=list(train_sizes_abs) + list(train_sizes_abs[::-1]),
            y=list(train_mean + train_std) + list((train_mean - train_std)[::-1]),
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Training ±1 std',
            showlegend=False
        ))
        
        # Validation score
        fig.add_trace(go.Scatter(
            x=train_sizes_abs,
            y=val_mean,
            mode='lines+markers',
            name='Validation Score',
            line=dict(color=self.colors['secondary']),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=list(train_sizes_abs) + list(train_sizes_abs[::-1]),
            y=list(val_mean + val_std) + list((val_mean - val_std)[::-1]),
            fill='toself',
            fillcolor='rgba(16, 185, 129, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Validation ±1 std',
            showlegend=False
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title='Training Set Size',
            yaxis_title='Score',
            template=self.theme,
            height=450,
            width=600,
            legend=dict(x=0.65, y=0.15)
        )
        
        return fig
    
    def plot_feature_correlation(
        self,
        X: np.ndarray,
        feature_names: List[str],
        method: str = 'pearson',
        title: str = "Feature Correlation Matrix"
    ):
        """
        Feature correlation heatmap зурах.
        
        Параметрүүд:
            X: Features
            feature_names: Feature нэрс
            method: Correlation method ('pearson', 'spearman', 'kendall')
            title: Графикийн гарчиг
            
        Буцаах:
            Plotly figure
        """
        import plotly.graph_objects as go
        
        df = pd.DataFrame(X, columns=feature_names)
        corr_matrix = df.corr(method=method)
        
        # Mask upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_masked = corr_matrix.where(~mask)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=feature_names,
            y=feature_names,
            colorscale='RdBu_r',
            zmid=0,
            showscale=True,
            colorbar=dict(title='Correlation')
        ))
        
        fig.update_layout(
            title=dict(text=f'{title} ({method.capitalize()})', x=0.5),
            template=self.theme,
            height=max(400, len(feature_names) * 20),
            width=max(500, len(feature_names) * 20),
            xaxis=dict(tickangle=45)
        )
        
        return fig
    
    def plot_threshold_analysis(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: np.ndarray = None,
        title: str = "Threshold Analysis"
    ):
        """
        Threshold өөрчлөгдөхөд precision, recall, F1 хэрхэн өөрчлөгдөхийг харуулна.
        
        Параметрүүд:
            y_true: Бодит label
            y_proba: Таамагласан магадлал
            thresholds: Шалгах threshold-ууд
            title: Графикийн гарчиг
            
        Буцаах:
            Plotly figure
        """
        import plotly.graph_objects as go
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 50)
        
        precisions = []
        recalls = []
        f1_scores = []
        
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        
        # Optimal F1 threshold
        optimal_idx = np.argmax(f1_scores)
        optimal_thresh = thresholds[optimal_idx]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=precisions,
            mode='lines',
            name='Precision',
            line=dict(color=self.colors['primary'])
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=recalls,
            mode='lines',
            name='Recall',
            line=dict(color=self.colors['secondary'])
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=f1_scores,
            mode='lines',
            name='F1 Score',
            line=dict(color=self.colors['warning'], width=2)
        ))
        
        # Optimal threshold marker
        fig.add_vline(
            x=optimal_thresh,
            line_dash='dash',
            line_color=self.colors['danger'],
            annotation_text=f'Optimal: {optimal_thresh:.3f}'
        )
        
        fig.update_layout(
            title=dict(
                text=f'{title}<br><sub>Optimal Threshold = {optimal_thresh:.3f} (F1 = {f1_scores[optimal_idx]:.3f})</sub>',
                x=0.5
            ),
            xaxis_title='Classification Threshold',
            yaxis_title='Score',
            template=self.theme,
            height=450,
            width=600,
            legend=dict(x=0.75, y=0.95)
        )
        
        return fig
    
    def plot_prediction_distribution(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "Prediction Probability Distribution"
    ):
        """
        Ангиар тусгаарлагдсан таамаглалын магадлалын хуваарилалт.
        
        Параметрүүд:
            y_true: Бодит label
            y_proba: Таамагласан магадлал
            title: Графикийн гарчиг
            
        Буцаах:
            Plotly figure
        """
        import plotly.graph_objects as go
        
        pos_proba = y_proba[y_true == 1]
        neg_proba = y_proba[y_true == 0]
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=neg_proba,
            name='Negative Class',
            opacity=0.7,
            marker_color=self.colors['primary'],
            nbinsx=30
        ))
        
        fig.add_trace(go.Histogram(
            x=pos_proba,
            name='Positive Class',
            opacity=0.7,
            marker_color=self.colors['danger'],
            nbinsx=30
        ))
        
        # Threshold line
        fig.add_vline(
            x=0.5,
            line_dash='dash',
            line_color='gray',
            annotation_text='Threshold (0.5)'
        )
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title='Predicted Probability',
            yaxis_title='Count',
            template=self.theme,
            height=400,
            width=600,
            barmode='overlay',
            legend=dict(x=0.7, y=0.95)
        )
        
        return fig
    
    def create_model_dashboard(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        class_names: Optional[List[str]] = None
    ):
        """
        Загварын гүйцэтгэлийн бүрэн dashboard үүсгэх.
        
        Параметрүүд:
            y_true: Бодит label
            y_pred: Таамагласан label
            y_proba: Таамагласан магадлал
            class_names: Ангийн нэрс
            
        Буцаах:
            Plotly figure (subplots)
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
        from sklearn.metrics import confusion_matrix, classification_report
        
        # Calculate metrics
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        cm = confusion_matrix(y_true, y_pred)
        
        if class_names is None:
            class_names = ['Negative', 'Positive']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'ROC Curve (AUC={auc:.3f})',
                'Precision-Recall Curve',
                'Confusion Matrix',
                'Prediction Distribution'
            )
        )
        
        # ROC Curve
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC',
                      line=dict(color=self.colors['primary'])),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                      line=dict(color='gray', dash='dash')),
            row=1, col=1
        )
        
        # PR Curve
        fig.add_trace(
            go.Scatter(x=recall, y=precision, mode='lines', name='PR',
                      line=dict(color=self.colors['secondary'])),
            row=1, col=2
        )
        
        # Confusion Matrix
        fig.add_trace(
            go.Heatmap(z=cm, x=class_names, y=class_names,
                      colorscale='Blues', showscale=False),
            row=2, col=1
        )
        
        # Prediction Distribution
        fig.add_trace(
            go.Histogram(x=y_proba[y_true == 0], name='Negative',
                        opacity=0.7, marker_color=self.colors['primary']),
            row=2, col=2
        )
        fig.add_trace(
            go.Histogram(x=y_proba[y_true == 1], name='Positive',
                        opacity=0.7, marker_color=self.colors['danger']),
            row=2, col=2
        )
        
        fig.update_layout(
            height=700,
            width=900,
            title_text='Model Performance Dashboard',
            showlegend=False,
            template=self.theme
        )
        
        return fig
