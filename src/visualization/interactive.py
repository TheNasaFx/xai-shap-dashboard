"""
Interactive Plots - Enhanced Interactive Visualizations
========================================================

Specialized interactive visualization components using
callbacks and dynamic updates.

Author: XAI-SHAP Framework
"""

import logging
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class InteractivePlots:
    """
    Interactive visualization components for dashboard integration.
    
    Provides components with:
    - Click handlers for sample selection
    - Range sliders for filtering
    - Feature selection dropdowns
    - Real-time updates
    
    Example:
        >>> interactive = InteractivePlots()
        >>> fig = interactive.create_interactive_summary(
        ...     shap_values, X, feature_names,
        ...     on_sample_click=handle_click
        ... )
    """
    
    def __init__(self, config=None):
        """Initialize InteractivePlots."""
        self.config = config
        self._callbacks = {}
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for interactive event."""
        self._callbacks[event] = callback
    
    def create_interactive_summary(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        max_display: int = 20
    ):
        """
        Create interactive summary plot with sample selection.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Calculate importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_order = np.argsort(mean_abs_shap)[::-1][:max_display]
        
        fig = go.Figure()
        
        for i, idx in enumerate(reversed(feature_order)):
            feature = feature_names[idx]
            values = shap_values[:, idx]
            feature_vals = X[:, idx]
            
            feat_norm = (feature_vals - feature_vals.min()) / (feature_vals.max() - feature_vals.min() + 1e-10)
            
            fig.add_trace(go.Scatter(
                x=values,
                y=np.random.normal(i, 0.1, len(values)),
                mode='markers',
                marker=dict(
                    size=6,
                    color=feat_norm,
                    colorscale='RdBu_r',
                    opacity=0.7
                ),
                name=feature,
                customdata=np.column_stack([
                    np.arange(len(values)),  # Sample indices
                    feature_vals
                ]),
                hovertemplate=(
                    f'{feature}<br>'
                    'Sample: %{customdata[0]}<br>'
                    'SHAP: %{x:.3f}<br>'
                    'Value: %{customdata[1]:.2f}'
                    '<extra></extra>'
                )
            ))
        
        fig.update_layout(
            title='Interactive SHAP Summary (click to select sample)',
            xaxis_title='SHAP Value',
            yaxis=dict(
                ticktext=[feature_names[i] for i in reversed(feature_order)],
                tickvals=list(range(len(feature_order)))
            ),
            height=max(500, len(feature_order) * 25),
            showlegend=False,
            template='plotly_white',
            clickmode='event+select'
        )
        
        return fig
    
    def create_feature_explorer(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str]
    ):
        """
        Create interactive feature exploration dashboard.
        
        Includes:
        - Feature selector dropdown
        - Dependence plot
        - Distribution histogram
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Default to most important feature
        importance = np.abs(shap_values).mean(axis=0)
        default_idx = np.argmax(importance)
        default_feature = feature_names[default_idx]
        
        # Create subplot layout
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                f'{default_feature} Dependence',
                f'{default_feature} SHAP Distribution'
            ]
        )
        
        # Dependence plot
        fig.add_trace(
            go.Scatter(
                x=X[:, default_idx],
                y=shap_values[:, default_idx],
                mode='markers',
                marker=dict(
                    color=shap_values[:, default_idx],
                    colorscale='RdBu_r',
                    size=6
                ),
                name='Values'
            ),
            row=1, col=1
        )
        
        # SHAP distribution histogram
        fig.add_trace(
            go.Histogram(
                x=shap_values[:, default_idx],
                nbinsx=30,
                name='SHAP Distribution',
                marker_color='steelblue'
            ),
            row=1, col=2
        )
        
        # Add dropdown for feature selection
        buttons = []
        for i, feature in enumerate(feature_names):
            buttons.append(dict(
                method='update',
                label=feature,
                args=[
                    {
                        'x': [X[:, i], shap_values[:, i]],
                        'y': [shap_values[:, i], None],
                        'marker.color': [shap_values[:, i], None]
                    },
                    {
                        'annotations[0].text': f'{feature} Dependence',
                        'annotations[1].text': f'{feature} SHAP Distribution'
                    }
                ]
            ))
        
        fig.update_layout(
            updatemenus=[dict(
                active=default_idx,
                buttons=buttons,
                direction='down',
                showactive=True,
                x=0.1,
                y=1.15
            )],
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        fig.update_xaxes(title_text='Feature Value', row=1, col=1)
        fig.update_yaxes(title_text='SHAP Value', row=1, col=1)
        fig.update_xaxes(title_text='SHAP Value', row=1, col=2)
        fig.update_yaxes(title_text='Count', row=1, col=2)
        
        return fig
    
    def create_prediction_explorer(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        predictions: np.ndarray,
        base_value: float = 0
    ):
        """
        Create interactive prediction explorer.
        
        Allows selecting samples to see detailed explanations.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        n_samples = min(100, len(predictions))
        sample_idx = np.random.choice(len(predictions), n_samples, replace=False)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Predictions Overview', 'Selected Sample Explanation'],
            row_heights=[0.3, 0.7],
            vertical_spacing=0.15
        )
        
        # Predictions scatter
        fig.add_trace(
            go.Scatter(
                x=np.arange(n_samples),
                y=predictions[sample_idx],
                mode='markers',
                marker=dict(
                    size=8,
                    color=predictions[sample_idx],
                    colorscale='Viridis',
                    showscale=True
                ),
                customdata=sample_idx,
                hovertemplate='Sample: %{customdata}<br>Prediction: %{y:.3f}',
                name='Predictions'
            ),
            row=1, col=1
        )
        
        # Default waterfall for first sample
        sample_shap = shap_values[sample_idx[0]]
        sorted_idx = np.argsort(np.abs(sample_shap))[::-1][:10]
        
        features = [feature_names[i] for i in sorted_idx]
        values = [sample_shap[i] for i in sorted_idx]
        
        fig.add_trace(
            go.Waterfall(
                orientation='h',
                y=['Base'] + features + ['Prediction'],
                x=[base_value] + values + [0],
                measure=['absolute'] + ['relative'] * len(values) + ['total'],
                connector={'mode': 'between'},
                decreasing={'marker': {'color': '#EF553B'}},
                increasing={'marker': {'color': '#636EFA'}},
                totals={'marker': {'color': '#00CC96'}}
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=700,
            template='plotly_white',
            showlegend=False,
            title='Prediction Explorer (click sample to see explanation)'
        )
        
        return fig
    
    def create_comparison_view(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        sample_indices: List[int]
    ):
        """
        Create side-by-side comparison of multiple predictions.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        n_samples = min(4, len(sample_indices))
        
        fig = make_subplots(
            rows=1, cols=n_samples,
            subplot_titles=[f'Sample {i}' for i in sample_indices[:n_samples]]
        )
        
        # Get common feature order based on mean importance
        mean_shap = np.abs(shap_values[sample_indices[:n_samples]]).mean(axis=0)
        feature_order = np.argsort(mean_shap)[::-1][:10]
        
        for col, idx in enumerate(sample_indices[:n_samples], 1):
            sample_shap = shap_values[idx]
            
            fig.add_trace(
                go.Bar(
                    x=sample_shap[feature_order],
                    y=[feature_names[i] for i in feature_order],
                    orientation='h',
                    marker_color=['#EF553B' if v < 0 else '#636EFA' 
                                  for v in sample_shap[feature_order]]
                ),
                row=1, col=col
            )
        
        fig.update_layout(
            height=400,
            template='plotly_white',
            showlegend=False,
            title='Sample Comparison View'
        )
        
        return fig
