"""
Explanation Types - Data Classes for Different Explanation Types
=================================================================

Structured data classes for representing different types of
SHAP explanations.

Author: XAI-SHAP Framework
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class LocalExplanation:
    """
    Local explanation for a single prediction.
    
    Represents how each feature contributed to a specific
    prediction for one sample.
    
    Attributes:
        sample_index: Index of the explained sample
        base_value: Expected model output (baseline)
        predicted_value: Actual model prediction
        feature_contributions: Dict mapping features to SHAP values
        feature_values: Dict mapping features to input values
        top_positive: Features pushing prediction higher
        top_negative: Features pushing prediction lower
    """
    sample_index: int
    base_value: float
    predicted_value: float
    feature_contributions: Dict[str, float]
    feature_values: Dict[str, float] = field(default_factory=dict)
    top_positive: List[Dict[str, float]] = field(default_factory=list)
    top_negative: List[Dict[str, float]] = field(default_factory=list)
    
    @property
    def total_contribution(self) -> float:
        """Sum of all feature contributions."""
        return sum(self.feature_contributions.values())
    
    def get_top_features(self, n: int = 5) -> List[str]:
        """Get top N most important features by absolute contribution."""
        sorted_features = sorted(
            self.feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return [f[0] for f in sorted_features[:n]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'sample_index': self.sample_index,
            'base_value': self.base_value,
            'predicted_value': self.predicted_value,
            'feature_contributions': self.feature_contributions,
            'feature_values': self.feature_values,
            'top_positive': self.top_positive,
            'top_negative': self.top_negative
        }
    
    def to_natural_language(self) -> str:
        """Generate human-readable explanation."""
        lines = [f"Prediction explanation for sample {self.sample_index}:"]
        lines.append(f"Base value (average prediction): {self.base_value:.4f}")
        lines.append(f"Final prediction: {self.predicted_value:.4f}")
        lines.append("\nTop contributing features:")
        
        for feature in self.get_top_features(5):
            contribution = self.feature_contributions[feature]
            direction = "↑" if contribution > 0 else "↓"
            value = self.feature_values.get(feature, "N/A")
            lines.append(f"  {direction} {feature}: {contribution:+.4f} (value: {value})")
        
        return "\n".join(lines)


@dataclass
class GlobalExplanation:
    """
    Global explanation for overall model behavior.
    
    Represents the general feature importance across
    all predictions.
    
    Attributes:
        feature_importance: Dict mapping features to importance scores
        mean_shap_values: Mean SHAP values per feature
        std_shap_values: Standard deviation of SHAP values
        n_samples: Number of samples used for computation
        feature_names: List of feature names in order
    """
    feature_importance: Dict[str, float]
    mean_shap_values: Dict[str, float]
    std_shap_values: Dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    feature_names: List[str] = field(default_factory=list)
    
    @property
    def ranked_features(self) -> List[str]:
        """Get features ranked by importance."""
        return sorted(
            self.feature_importance.keys(),
            key=lambda x: self.feature_importance[x],
            reverse=True
        )
    
    def get_top_features(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N most important features with details."""
        ranked = self.ranked_features[:n]
        return [
            {
                'feature': f,
                'importance': self.feature_importance[f],
                'mean_shap': self.mean_shap_values.get(f, 0),
                'std_shap': self.std_shap_values.get(f, 0)
            }
            for f in ranked
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'feature_importance': self.feature_importance,
            'mean_shap_values': self.mean_shap_values,
            'std_shap_values': self.std_shap_values,
            'n_samples': self.n_samples,
            'ranked_features': self.ranked_features
        }
    
    def to_natural_language(self) -> str:
        """Generate human-readable summary."""
        lines = ["Global Feature Importance Summary:"]
        lines.append(f"Based on analysis of {self.n_samples} samples\n")
        lines.append("Top important features:")
        
        for i, item in enumerate(self.get_top_features(10), 1):
            lines.append(
                f"  {i}. {item['feature']}: "
                f"importance={item['importance']:.4f}, "
                f"mean SHAP={item['mean_shap']:.4f}"
            )
        
        return "\n".join(lines)


@dataclass
class InteractionExplanation:
    """
    Feature interaction explanation.
    
    Represents interactions between pairs of features
    in model predictions.
    
    Attributes:
        feature1: First feature name
        feature2: Second feature name
        interaction_values: Array of interaction SHAP values
        strength: Overall interaction strength
        direction: 'synergistic' or 'antagonistic'
    """
    feature1: str
    feature2: str
    interaction_values: np.ndarray
    strength: float = 0.0
    direction: str = 'unknown'
    
    @classmethod
    def from_shap_interaction(
        cls,
        feature1: str,
        feature2: str,
        interaction_values: np.ndarray
    ) -> 'InteractionExplanation':
        """Create from SHAP interaction values."""
        strength = np.abs(interaction_values).mean()
        
        # Determine direction based on correlation with main effects
        mean_interaction = interaction_values.mean()
        direction = 'synergistic' if mean_interaction > 0 else 'antagonistic'
        
        return cls(
            feature1=feature1,
            feature2=feature2,
            interaction_values=interaction_values,
            strength=strength,
            direction=direction
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'feature1': self.feature1,
            'feature2': self.feature2,
            'strength': self.strength,
            'direction': self.direction,
            'mean_interaction': float(self.interaction_values.mean()),
            'std_interaction': float(self.interaction_values.std())
        }
    
    def to_natural_language(self) -> str:
        """Generate human-readable description."""
        return (
            f"Interaction between '{self.feature1}' and '{self.feature2}':\n"
            f"  Strength: {self.strength:.4f}\n"
            f"  Direction: {self.direction}\n"
            f"  This indicates that the effect of {self.feature1} on predictions "
            f"{'increases' if self.direction == 'synergistic' else 'decreases'} "
            f"when {self.feature2} is present."
        )


def create_local_explanations(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: List[str],
    base_value: float,
    predictions: Optional[np.ndarray] = None
) -> List[LocalExplanation]:
    """
    Factory function to create LocalExplanation objects.
    
    Args:
        shap_values: SHAP values array (n_samples, n_features)
        feature_values: Input feature values
        feature_names: List of feature names
        base_value: Expected model output
        predictions: Optional actual predictions
        
    Returns:
        List of LocalExplanation objects
    """
    explanations = []
    
    for i in range(len(shap_values)):
        contributions = dict(zip(feature_names, shap_values[i].tolist()))
        values = dict(zip(feature_names, feature_values[i].tolist()))
        
        predicted = predictions[i] if predictions is not None else base_value + shap_values[i].sum()
        
        # Get top positive and negative contributors
        sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        top_pos = [{'feature': f, 'contribution': v} for f, v in sorted_contrib if v > 0][:5]
        top_neg = [{'feature': f, 'contribution': v} for f, v in sorted_contrib if v < 0][-5:]
        
        explanations.append(LocalExplanation(
            sample_index=i,
            base_value=float(base_value),
            predicted_value=float(predicted),
            feature_contributions=contributions,
            feature_values=values,
            top_positive=top_pos,
            top_negative=top_neg
        ))
    
    return explanations


def create_global_explanation(
    shap_values: np.ndarray,
    feature_names: List[str]
) -> GlobalExplanation:
    """
    Factory function to create GlobalExplanation object.
    
    Args:
        shap_values: SHAP values array (n_samples, n_features)
        feature_names: List of feature names
        
    Returns:
        GlobalExplanation object
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)
    std_shap = shap_values.std(axis=0)
    
    return GlobalExplanation(
        feature_importance=dict(zip(feature_names, mean_abs_shap.tolist())),
        mean_shap_values=dict(zip(feature_names, mean_shap.tolist())),
        std_shap_values=dict(zip(feature_names, std_shap.tolist())),
        n_samples=len(shap_values),
        feature_names=feature_names
    )
