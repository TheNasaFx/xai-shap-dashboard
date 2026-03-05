"""
Plot Themes - Visualization Theme Management
=============================================

Manages consistent styling across all visualizations.

Author: XAI-SHAP Framework
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlotTheme:
    """
    Theme configuration for visualizations.
    
    Defines colors, fonts, and styling for consistent
    appearance across all plots.
    """
    name: str = "default"
    
    # Colors
    primary_color: str = "#636EFA"
    secondary_color: str = "#EF553B"
    positive_color: str = "#00CC96"
    negative_color: str = "#EF553B"
    neutral_color: str = "#636EFA"
    background_color: str = "#FFFFFF"
    grid_color: str = "#E5E5E5"
    
    # Color scales
    diverging_colorscale: str = "RdBu_r"
    sequential_colorscale: str = "Blues"
    categorical_colors: List[str] = field(default_factory=lambda: [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", 
        "#FFA15A", "#19D3F3", "#FF6692", "#B6E880"
    ])
    
    # Fonts
    font_family: str = "Arial, sans-serif"
    title_font_size: int = 16
    label_font_size: int = 12
    tick_font_size: int = 10
    
    # Layout
    margin: Dict[str, int] = field(default_factory=lambda: {
        'l': 60, 'r': 40, 't': 60, 'b': 60
    })
    
    def to_plotly_template(self) -> dict:
        """Convert theme to Plotly template format."""
        return {
            'layout': {
                'font': {
                    'family': self.font_family,
                    'size': self.label_font_size
                },
                'title': {
                    'font': {'size': self.title_font_size}
                },
                'paper_bgcolor': self.background_color,
                'plot_bgcolor': self.background_color,
                'colorway': self.categorical_colors,
                'xaxis': {
                    'gridcolor': self.grid_color,
                    'tickfont': {'size': self.tick_font_size}
                },
                'yaxis': {
                    'gridcolor': self.grid_color,
                    'tickfont': {'size': self.tick_font_size}
                },
                'margin': self.margin
            }
        }


class ThemeManager:
    """
    Manage visualization themes.
    
    Provides preset themes and custom theme creation.
    """
    
    _themes: Dict[str, PlotTheme] = {}
    _current_theme: str = "default"
    
    @classmethod
    def register_theme(cls, theme: PlotTheme) -> None:
        """Register a theme."""
        cls._themes[theme.name] = theme
    
    @classmethod
    def get_theme(cls, name: Optional[str] = None) -> PlotTheme:
        """Get theme by name."""
        name = name or cls._current_theme
        if name not in cls._themes:
            cls._initialize_default_themes()
        return cls._themes.get(name, cls._themes.get('default', PlotTheme()))
    
    @classmethod
    def set_current_theme(cls, name: str) -> None:
        """Set current theme."""
        if name in cls._themes:
            cls._current_theme = name
        else:
            logger.warning(f"Theme '{name}' not found. Using default.")
    
    @classmethod
    def list_themes(cls) -> List[str]:
        """List available themes."""
        cls._initialize_default_themes()
        return list(cls._themes.keys())
    
    @classmethod
    def _initialize_default_themes(cls):
        """Initialize default themes."""
        if cls._themes:
            return
        
        # Default theme
        cls._themes['default'] = PlotTheme(name='default')
        
        # Dark theme
        cls._themes['dark'] = PlotTheme(
            name='dark',
            background_color='#1e1e1e',
            grid_color='#3e3e3e',
            primary_color='#5c9eff',
            secondary_color='#ff6b6b'
        )
        
        # Minimal theme
        cls._themes['minimal'] = PlotTheme(
            name='minimal',
            grid_color='#f0f0f0',
            title_font_size=14,
            label_font_size=11
        )
        
        # Colorful theme
        cls._themes['colorful'] = PlotTheme(
            name='colorful',
            diverging_colorscale='Spectral',
            sequential_colorscale='Viridis',
            categorical_colors=[
                '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
                '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'
            ]
        )
        
        # Professional theme
        cls._themes['professional'] = PlotTheme(
            name='professional',
            primary_color='#2C3E50',
            secondary_color='#E74C3C',
            positive_color='#27AE60',
            negative_color='#C0392B',
            font_family='Helvetica Neue, Helvetica, Arial, sans-serif'
        )


# Initialize default themes
ThemeManager._initialize_default_themes()
