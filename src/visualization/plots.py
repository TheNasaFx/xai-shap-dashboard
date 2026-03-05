"""
XAI Visualizer - Визуализацийн үндсэн хөдөлгүүр
=================================================

SHAP тайлбаруудад зориулсан иж бүрэн визуализацийн
боломжуудыг Plotly-г ашиглан интерактив график,
Matplotlib/SHAP-г ашиглан статик график үүсгэнэ.

Зохиогч: XAI-SHAP Framework
"""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class XAIVisualizer:
    """
    XAI-SHAP framework-ийн үндсэн визуализацийн класс.
    
    Энэ класс дараах боломжуудыг олгоно:
    - Plotly ашиглан интерактив SHAP визуализаци
    - matplotlib ашиглан статик SHAP график
    - Тайланд зориулсан тусгай загварчилсан график
    - Dashboard-д бэлэн компонентууд
    
    Боломжит график төрлүүд:
    - 'summary': SHAP summary plot (beeswarm)
    - 'bar': Feature importance баганан график
    - 'waterfall': Ганц таамаглалд waterfall plot
    - 'force': Таамаглалын тайлбарт force plot
    - 'dependence': Feature dependence plot
    - 'heatmap': SHAP утгуудын heatmap
    - 'violin': SHAP хуваарилалтын violin plot
    - 'scatter': SHAP өнгөтэй scatter plot
    
    Жишээ:
        >>> visualizer = XAIVisualizer(config)
        >>> fig = visualizer.plot(
        ...     plot_type='summary',
        ...     shap_values=shap_values,
        ...     X=X_test,
        ...     feature_names=feature_names
        ... )
        >>> fig.show()
    """
    
    def __init__(self, config=None):
        """
        XAI Visualizer эхлүүлэх.
        
        Параметрүүд:
            config: Тохиргооны менежер instance
        """
        self.config = config
        
        # Тохиргооны тохируулгуудыг авах
        if config:
            self.default_theme = config.get("visualization.default_theme", "plotly")
            self.interactive = config.get("visualization.interactive", True)
            self.fig_width = config.get("visualization.figure_width", 10)
            self.fig_height = config.get("visualization.figure_height", 6)
            self.color_palette = config.get("visualization.color_palette", "viridis")
        else:
            self.default_theme = "plotly"
            self.interactive = True
            self.fig_width = 10
            self.fig_height = 6
            self.color_palette = "viridis"
    
    def plot(
        self,
        plot_type: str,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        **kwargs
    ) -> Any:
        """
        График төрлөөс хамаарч визуализаци үүсгэх.
        
        Параметрүүд:
            plot_type: Үүсгэх графикийн төрөл
            shap_values: SHAP утгуудын массив
            X: Feature утгуудын массив
            feature_names: Feature нэрсийн жагсаалт
            **kwargs: Нэмэлт график-тодорхой параметрүүд
            
        Буцаах:
            Plotly figure эсвэл matplotlib axes
        """
        plot_methods = {
            'summary': self._plot_summary,
            'bar': self._plot_bar,
            'waterfall': self._plot_waterfall,
            'force': self._plot_force,
            'dependence': self._plot_dependence,
            'heatmap': self._plot_heatmap,
            'violin': self._plot_violin,
            'scatter': self._plot_scatter,
            'beeswarm': self._plot_beeswarm
        }
        
        if plot_type not in plot_methods:
            raise ValueError(f"Тодорхойгүй график төрөл: {plot_type}. Боломжит: {list(plot_methods.keys())}")
        
        return plot_methods[plot_type](
            shap_values=shap_values,
            X=X,
            feature_names=feature_names,
            **kwargs
        )
    
    def _plot_summary(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        max_display: int = 20,
        **kwargs
    ):
        """
        Интерактив summary plot (beeswarm загвар) үүсгэх.
        
        Feature бүрийн SHAP утгуудын хуваарилалтыг харуулна.
        """
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Өгөгдлийг бэлтгэх
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_order = np.argsort(mean_abs_shap)[::-1][:max_display]
        
        # Figure үүсгэх
        fig = go.Figure()
        
        for i, idx in enumerate(reversed(feature_order)):
            feature = feature_names[idx]
            values = shap_values[:, idx]
            feature_vals = X[:, idx]
            
            # Feature утгуудыг өнгөнд normalize хийх
            feat_norm = (feature_vals - feature_vals.min()) / (feature_vals.max() - feature_vals.min() + 1e-10)
            
            fig.add_trace(go.Scatter(
                x=values,
                y=np.random.normal(i, 0.1, len(values)),
                mode='markers',
                marker=dict(
                    size=5,
                    color=feat_norm,
                    colorscale='RdBu_r',
                    opacity=0.6
                ),
                name=feature,
                hovertemplate=f'{feature}<br>SHAP: %{{x:.3f}}<br>Утга: %{{customdata:.2f}}',
                customdata=feature_vals
            ))
        
        fig.update_layout(
            title='SHAP Summary Plot - Шинж чанаруудын нөлөөллийн график',
            xaxis_title='SHAP утга (загварын гаралтад үзүүлэх нөлөө)',
            yaxis=dict(
                ticktext=[feature_names[i] for i in reversed(feature_order)],
                tickvals=list(range(len(feature_order))),
                title='Шинж чанарууд (Features)'
            ),
            height=max(400, len(feature_order) * 25),
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def _plot_bar(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        max_display: int = 20,
        **kwargs
    ):
        """
        Feature importance баганан график үүсгэх.
        
        Feature бүрийн дундаж абсолют SHAP утгыг харуулна.
        """
        import plotly.express as px
        
        # Дундаж абсолют SHAP утгыг тооцоолох
        importance = np.abs(shap_values).mean(axis=0)
        
        # DataFrame үүсгэх
        df = pd.DataFrame({
            'Шинж чанар': feature_names,
            'Ач холбогдол': importance
        }).sort_values('Ач холбогдол', ascending=True).tail(max_display)
        
        fig = px.bar(
            df,
            x='Ач холбогдол',
            y='Шинж чанар',
            orientation='h',
            title='Шинж чанаруудын ач холбогдол (Дундаж |SHAP утга|)',
            labels={'Ач холбогдол': 'Дундаж |SHAP утга|'},
            color='Ач холбогдол',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            height=max(400, len(df) * 25),
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def _plot_waterfall(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        sample_idx: int = 0,
        base_value: float = 0,
        max_display: int = 15,
        **kwargs
    ):
        """
        Create waterfall plot for single prediction explanation.
        
        Shows how each feature contributes to moving from
        base value to final prediction.
        """
        import plotly.graph_objects as go
        
        # Get single sample
        sample_shap = shap_values[sample_idx]
        sample_X = X[sample_idx]
        
        # Sort by absolute SHAP value
        sorted_indices = np.argsort(np.abs(sample_shap))[::-1][:max_display]
        
        # Prepare data
        features = [feature_names[i] for i in sorted_indices]
        values = [sample_shap[i] for i in sorted_indices]
        feature_vals = [sample_X[i] for i in sorted_indices]
        
        # Calculate cumulative values
        cumsum = np.cumsum([base_value] + values)
        
        # Create waterfall
        fig = go.Figure(go.Waterfall(
            orientation='v',
            measure=['absolute'] + ['relative'] * len(values) + ['total'],
            x=['Base'] + [f'{f}\n({v:.2f})' for f, v in zip(features, feature_vals)] + ['Prediction'],
            y=[base_value] + values + [0],
            connector={'line': {'color': 'rgb(63, 63, 63)'}},
            decreasing={'marker': {'color': '#EF553B'}},
            increasing={'marker': {'color': '#636EFA'}},
            totals={'marker': {'color': '#00CC96'}}
        ))
        
        fig.update_layout(
            title=f'Waterfall Plot - Sample {sample_idx}',
            yaxis_title='Model Output',
            height=500,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def _plot_force(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        sample_idx: int = 0,
        base_value: float = 0,
        **kwargs
    ):
        """
        Create force plot visualization.
        
        Shows features pushing prediction higher (red) and
        lower (blue) from the base value.
        """
        import plotly.graph_objects as go
        
        sample_shap = shap_values[sample_idx]
        sample_X = X[sample_idx]
        
        # Separate positive and negative contributions
        pos_idx = np.where(sample_shap > 0)[0]
        neg_idx = np.where(sample_shap < 0)[0]
        
        # Sort within each group
        pos_sorted = pos_idx[np.argsort(sample_shap[pos_idx])[::-1]]
        neg_sorted = neg_idx[np.argsort(sample_shap[neg_idx])]
        
        fig = go.Figure()
        
        # Starting position
        current = base_value
        final = base_value + sample_shap.sum()
        
        # Add base value
        fig.add_trace(go.Bar(
            x=[current],
            y=['Base Value'],
            orientation='h',
            marker_color='gray',
            name='Base',
            hovertemplate='Base Value: %{x:.3f}'
        ))
        
        # Add positive contributions
        for idx in pos_sorted[:5]:
            feature = feature_names[idx]
            value = sample_shap[idx]
            fig.add_trace(go.Bar(
                x=[value],
                y=[f'{feature}'],
                orientation='h',
                marker_color='#EF553B',
                name=feature,
                hovertemplate=f'{feature}: +%{{x:.3f}}'
            ))
        
        # Add negative contributions
        for idx in neg_sorted[:5]:
            feature = feature_names[idx]
            value = sample_shap[idx]
            fig.add_trace(go.Bar(
                x=[value],
                y=[f'{feature}'],
                orientation='h',
                marker_color='#636EFA',
                name=feature,
                hovertemplate=f'{feature}: %{{x:.3f}}'
            ))
        
        fig.update_layout(
            title=f'Force Plot - Sample {sample_idx} (Prediction: {final:.3f})',
            xaxis_title='SHAP Value',
            barmode='relative',
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def _plot_dependence(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        feature: Optional[str] = None,
        interaction_feature: Optional[str] = None,
        **kwargs
    ):
        """
        Create dependence plot for a specific feature.
        
        Shows relationship between feature value and its
        SHAP value, optionally colored by interaction feature.
        """
        import plotly.express as px
        
        # Select feature
        if feature is None:
            # Use most important feature
            importance = np.abs(shap_values).mean(axis=0)
            feature_idx = np.argmax(importance)
            feature = feature_names[feature_idx]
        else:
            feature_idx = feature_names.index(feature)
        
        # Prepare data
        df = pd.DataFrame({
            'Feature Value': X[:, feature_idx],
            'SHAP Value': shap_values[:, feature_idx]
        })
        
        # Add interaction coloring
        if interaction_feature:
            interact_idx = feature_names.index(interaction_feature)
            df['Interaction'] = X[:, interact_idx]
            fig = px.scatter(
                df,
                x='Feature Value',
                y='SHAP Value',
                color='Interaction',
                color_continuous_scale='RdBu_r',
                title=f'Dependence Plot: {feature} (colored by {interaction_feature})'
            )
        else:
            fig = px.scatter(
                df,
                x='Feature Value',
                y='SHAP Value',
                title=f'Dependence Plot: {feature}'
            )
        
        fig.update_layout(
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def _plot_heatmap(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        max_display: int = 20,
        n_samples: int = 50,
        **kwargs
    ):
        """
        Create heatmap of SHAP values.
        
        Shows SHAP values for multiple samples across features.
        """
        import plotly.express as px
        
        # Select top features
        importance = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(importance)[::-1][:max_display]
        
        # Select samples
        if len(shap_values) > n_samples:
            sample_idx = np.random.choice(len(shap_values), n_samples, replace=False)
        else:
            sample_idx = np.arange(len(shap_values))
        
        # Create DataFrame
        df = pd.DataFrame(
            shap_values[sample_idx][:, top_idx],
            columns=[feature_names[i] for i in top_idx]
        )
        
        fig = px.imshow(
            df.T,
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0,
            title='SHAP Values Heatmap',
            labels={'x': 'Sample', 'y': 'Feature', 'color': 'SHAP Value'}
        )
        
        fig.update_layout(
            height=max(400, len(top_idx) * 25),
            template='plotly_white'
        )
        
        return fig
    
    def _plot_violin(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        max_display: int = 15,
        **kwargs
    ):
        """
        Create violin plot of SHAP value distributions.
        """
        import plotly.graph_objects as go
        
        # Select top features
        importance = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(importance)[::-1][:max_display]
        
        fig = go.Figure()
        
        for idx in reversed(top_idx):
            feature = feature_names[idx]
            values = shap_values[:, idx]
            
            fig.add_trace(go.Violin(
                x=values,
                name=feature,
                box_visible=True,
                meanline_visible=True,
                opacity=0.7
            ))
        
        fig.update_layout(
            title='SHAP Value Distribution by Feature',
            xaxis_title='SHAP Value',
            height=max(400, len(top_idx) * 40),
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def _plot_scatter(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        x_feature: Optional[str] = None,
        y_feature: Optional[str] = None,
        **kwargs
    ):
        """
        Create scatter plot colored by SHAP values.
        """
        import plotly.express as px
        
        importance = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(importance)[::-1]
        
        if x_feature is None:
            x_idx = sorted_idx[0]
            x_feature = feature_names[x_idx]
        else:
            x_idx = feature_names.index(x_feature)
        
        if y_feature is None:
            y_idx = sorted_idx[1]
            y_feature = feature_names[y_idx]
        else:
            y_idx = feature_names.index(y_feature)
        
        df = pd.DataFrame({
            x_feature: X[:, x_idx],
            y_feature: X[:, y_idx],
            'SHAP Sum': shap_values.sum(axis=1)
        })
        
        fig = px.scatter(
            df,
            x=x_feature,
            y=y_feature,
            color='SHAP Sum',
            color_continuous_scale='RdBu_r',
            title=f'{x_feature} vs {y_feature} (colored by total SHAP)'
        )
        
        fig.update_layout(
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def _plot_beeswarm(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        **kwargs
    ):
        """Alias for summary plot."""
        return self._plot_summary(shap_values, X, feature_names, **kwargs)
    
    def plot_multiple(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        plot_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create multiple plots at once.
        
        Args:
            shap_values: SHAP values array
            X: Feature values array
            feature_names: List of feature names
            plot_types: List of plot types to create
            
        Returns:
            Dictionary mapping plot type to figure
        """
        if plot_types is None:
            plot_types = ['summary', 'bar', 'waterfall']
        
        figures = {}
        for pt in plot_types:
            try:
                figures[pt] = self.plot(
                    plot_type=pt,
                    shap_values=shap_values,
                    X=X,
                    feature_names=feature_names
                )
            except Exception as e:
                logger.warning(f"Failed to create {pt} plot: {e}")
        
        return figures
