"""
Bias Detector - Data Bias Detection Module
===========================================

Detects potential bias in datasets based on protected attributes.
Essential for Responsible AI implementation.

Author: XAI-SHAP Framework
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BiasDetector:
    """
    Detect potential bias in datasets.
    
    This class analyzes data for potential bias related to protected
    attributes such as gender, race, age, etc.
    
    Bias detection methods:
    - Class imbalance analysis
    - Representation disparity
    - Label distribution across groups
    - Statistical parity checks
    
    Example:
        >>> detector = BiasDetector()
        >>> report = detector.detect(df, protected_attrs=['gender', 'age_group'])
        >>> print(report['summary'])
    """
    
    def __init__(self, threshold: float = 0.8):
        """
        Initialize BiasDetector.
        
        Args:
            threshold: Ratio threshold for detecting imbalance (default 0.8)
        """
        self.threshold = threshold
        self._reports = {}
    
    def detect(
        self,
        data: pd.DataFrame,
        protected_attributes: List[str],
        target: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect bias in the dataset.
        
        Args:
            data: Input DataFrame
            protected_attributes: List of protected attribute column names
            target: Optional target column for label distribution analysis
            
        Returns:
            Bias detection report
        """
        logger.info(f"Running bias detection on {len(protected_attributes)} protected attributes")
        
        report = {
            'has_bias': False,
            'protected_attributes': protected_attributes,
            'findings': [],
            'metrics': {},
            'recommendations': []
        }
        
        for attr in protected_attributes:
            if attr not in data.columns:
                logger.warning(f"Protected attribute '{attr}' not found in data")
                continue
            
            # Analyze representation
            attr_report = self._analyze_attribute(data, attr, target)
            report['metrics'][attr] = attr_report
            
            if attr_report['has_imbalance']:
                report['has_bias'] = True
                report['findings'].append(
                    f"Imbalance detected in '{attr}': {attr_report['imbalance_description']}"
                )
        
        # Generate summary and recommendations
        report['summary'] = self._generate_summary(report)
        report['recommendations'] = self._generate_recommendations(report)
        
        self._reports = report
        
        return report
    
    def _analyze_attribute(
        self,
        data: pd.DataFrame,
        attribute: str,
        target: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze a single protected attribute for bias."""
        attr_data = data[attribute]
        
        # Value distribution
        value_counts = attr_data.value_counts(normalize=True)
        
        # Check for imbalance
        min_ratio = value_counts.min()
        max_ratio = value_counts.max()
        imbalance_ratio = min_ratio / max_ratio if max_ratio > 0 else 0
        
        has_imbalance = imbalance_ratio < self.threshold
        
        result = {
            'attribute': attribute,
            'n_unique_values': len(value_counts),
            'value_distribution': value_counts.to_dict(),
            'min_representation': min_ratio,
            'max_representation': max_ratio,
            'imbalance_ratio': imbalance_ratio,
            'has_imbalance': has_imbalance,
            'imbalance_description': ''
        }
        
        if has_imbalance:
            underrepresented = value_counts.idxmin()
            result['imbalance_description'] = (
                f"'{underrepresented}' is underrepresented "
                f"({min_ratio:.1%} vs {max_ratio:.1%})"
            )
        
        # If target provided, check label distribution across groups
        if target and target in data.columns:
            result['label_distribution'] = self._check_label_distribution(
                data, attribute, target
            )
        
        return result
    
    def _check_label_distribution(
        self,
        data: pd.DataFrame,
        attribute: str,
        target: str
    ) -> Dict[str, Any]:
        """Check target distribution across attribute groups."""
        cross_tab = pd.crosstab(data[attribute], data[target], normalize='index')
        
        # Calculate statistical parity difference
        if cross_tab.shape[1] == 2:  # Binary target
            positive_rates = cross_tab.iloc[:, 1]
            spd = positive_rates.max() - positive_rates.min()
        else:
            spd = None
        
        return {
            'cross_tabulation': cross_tab.to_dict(),
            'statistical_parity_difference': spd,
            'has_label_bias': spd > 0.1 if spd is not None else False
        }
    
    def _generate_summary(self, report: Dict[str, Any]) -> str:
        """Generate human-readable summary."""
        if not report['has_bias']:
            return "No significant bias detected in the dataset."
        
        n_issues = len(report['findings'])
        return (
            f"Potential bias detected: {n_issues} issue(s) found. "
            f"Review findings and consider mitigation strategies."
        )
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations = []
        
        if not report['has_bias']:
            recommendations.append("Continue monitoring for bias as data evolves.")
            return recommendations
        
        # Based on findings
        for attr, metrics in report['metrics'].items():
            if metrics.get('has_imbalance', False):
                recommendations.append(
                    f"Consider oversampling underrepresented groups in '{attr}' "
                    f"or using class weights during training."
                )
            
            if metrics.get('label_distribution', {}).get('has_label_bias', False):
                recommendations.append(
                    f"Apply fairness constraints during model training "
                    f"to ensure equitable outcomes across '{attr}' groups."
                )
        
        recommendations.append(
            "Use fairness-aware algorithms or post-processing techniques."
        )
        recommendations.append(
            "Document bias findings and mitigation steps for transparency."
        )
        
        return recommendations
    
    def get_detailed_report(self) -> pd.DataFrame:
        """Get detailed bias report as DataFrame."""
        if not self._reports:
            return pd.DataFrame()
        
        rows = []
        for attr, metrics in self._reports.get('metrics', {}).items():
            rows.append({
                'Attribute': attr,
                'Unique Values': metrics['n_unique_values'],
                'Min Representation': f"{metrics['min_representation']:.1%}",
                'Max Representation': f"{metrics['max_representation']:.1%}",
                'Imbalance Ratio': f"{metrics['imbalance_ratio']:.2f}",
                'Has Imbalance': metrics['has_imbalance']
            })
        
        return pd.DataFrame(rows)
