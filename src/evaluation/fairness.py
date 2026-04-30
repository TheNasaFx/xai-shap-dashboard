"""
Fairness Evaluation - Хариуцлагатай AI шударга байдлын хэмжигдэхүүнүүд
========================================================================

Хамгаалагдсан бүлгүүдэд тэгш хандлагыг баталгаажуулахын тулд
машин сургалтын загваруудад шударга байдлын үнэлгээг хангана.

Зохиогч: XAI-SHAP Framework
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils.helpers import predict_with_threshold

logger = logging.getLogger(__name__)


class BiasDetector:
    """
    Өгөгдлийн багц болон таамаглалуудад bias илрүүлэх.
    
    data_processing-оос дахин экспортлогдсон.
    """
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self._reports = {}
    
    def analyze_predictions(
        self,
        y_pred: np.ndarray,
        protected: np.ndarray
    ) -> Dict[str, Any]:
        """
        Хамгаалагдсан бүлгээр таамаглалуудыг шинжлэх.
        
        Параметрүүд:
            y_pred: Таамагласан шошго
            protected: Хамгаалагдсан шинж чанарын утгууд
            
        Буцаах:
            Бүлгийн статистиктай dictionary
        """
        groups = np.unique(protected)
        group_statistics = {}
        
        for group in groups:
            mask = protected == group
            group_pred = y_pred[mask]
            
            group_statistics[group] = {
                'тоо': int(mask.sum()),
                'эерэг_хувь': float(np.mean(group_pred)),
                'эерэг_тоо': int(np.sum(group_pred)),
                'сөрөг_тоо': int(np.sum(1 - group_pred))
            }
        
        return {
            'бүлгийн_статистик': group_statistics,
            'нийт_жишээ': len(y_pred),
            'ерөнхий_эерэг_хувь': float(np.mean(y_pred))
        }
    
    def has_significant_bias(
        self,
        y_pred: np.ndarray,
        protected: np.ndarray,
        threshold: float = 0.1
    ) -> bool:
        """
        Таамаглалуудад мэдэгдэхүйц bias байгаа эсэхийг шалгах.
        
        Параметрүүд:
            y_pred: Таамагласан шошго
            protected: Хамгаалагдсан шинж чанарын утгууд
            threshold: Эерэг хувьд зөвшөөрөгдөх хамгийн их ялгаа
            
        Буцаах:
            Мэдэгдэхүйц bias илэрсэн бол True
        """
        groups = np.unique(protected)
        rates = []
        
        for group in groups:
            mask = protected == group
            if mask.sum() > 0:
                rates.append(np.mean(y_pred[mask]))
        
        if len(rates) >= 2:
            return (max(rates) - min(rates)) > threshold
        
        return False
    
    def detect(
        self,
        data: pd.DataFrame,
        protected_attributes: List[str],
        target: Optional[str] = None
    ) -> Dict[str, Any]:
        """Өгөгдлийн багцад bias илрүүлэх."""
        report = {
            'has_bias': False,
            'хамгаалагдсан_шинж_чанарууд': protected_attributes,
            'олдворууд': [],
            'хэмжигдэхүүнүүд': {}
        }
        
        for attr in protected_attributes:
            if attr not in data.columns:
                continue
            
            value_counts = data[attr].value_counts(normalize=True)
            min_ratio = value_counts.min()
            max_ratio = value_counts.max()
            imbalance = min_ratio / max_ratio if max_ratio > 0 else 0
            
            has_imbalance = imbalance < self.threshold
            
            report['хэмжигдэхүүнүүд'][attr] = {
                'тэнцвэргүй_харьцаа': imbalance,
                'тэнцвэргүй': has_imbalance,
                'хуваарилалт': value_counts.to_dict()
            }
            
            if has_imbalance:
                report['has_bias'] = True
                report['олдворууд'].append(f"'{attr}' дахь тэнцвэргүй байдал")
        
        # 'summary' key-г ашиглах (framework.py-д хэрэглэгддэг)
        report['summary'] = (
            f"{len(report['олдворууд'])} шинж чанарт bias илэрлээ"
            if report['has_bias'] else "Мэдэгдэхүйц bias илрээгүй"
        )
        
        return report


class FairnessEvaluator:
    """
    Хамгаалагдсан бүлгүүдэд загварын шударга байдлыг үнэлэх.
    
    Тооцоолох шударга байдлын хэмжигдэхүүнүүд:
    - Demographic Parity: Бүлгүүд дээр тэнцүү эерэг хувь
    - Equalized Odds: Бүлгүүд дээр тэнцүү TPR болон FPR
    - Calibration: Таамагласан магадлал бодит үр дүнтэй тохирох
    - Disparate Impact: Эерэг хувийн харьцаа
    
    Жишээ:
        >>> evaluator = FairnessEvaluator()
        >>> results = evaluator.evaluate(
        ...     model, X_test, y_test,
        ...     protected_attributes=['gender']
        ... )
        >>> print(results['demographic_parity'])
    """
    
    def __init__(self, threshold: float = 0.8):
        """
        FairnessEvaluator эхлүүлэх.
        
        Параметрүүд:
            threshold: Шударга байдлын босго (жнь: 80%-ийн дүрмэнд 0.8)
        """
        self.threshold = threshold
    
    def demographic_parity(
        self,
        y_pred: np.ndarray,
        protected: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate demographic parity metrics.
        
        Args:
            y_pred: Predicted labels
            protected: Protected attribute values
            
        Returns:
            Dictionary with demographic parity metrics
        """
        groups = np.unique(protected)
        group_rates = {}
        
        for group in groups:
            mask = protected == group
            if mask.sum() > 0:
                rate = np.mean(y_pred[mask])
                group_rates[group] = float(rate)
        
        rates = list(group_rates.values())
        
        if len(rates) >= 2:
            dp_diff = max(rates) - min(rates)
            dp_ratio = min(rates) / max(rates) if max(rates) > 0 else 1.0
        else:
            dp_diff = 0.0
            dp_ratio = 1.0
        
        result = {
            'demographic_parity_difference': float(dp_diff),
            'demographic_parity_ratio': float(dp_ratio)
        }
        
        # Add group-specific rates
        for group, rate in group_rates.items():
            result[f'group_{group}_positive_rate'] = rate
        
        return result
    
    def disparate_impact(
        self,
        y_pred: np.ndarray,
        protected: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate disparate impact ratio.
        
        Args:
            y_pred: Predicted labels
            protected: Protected attribute values
            
        Returns:
            Dictionary with disparate impact metrics
        """
        groups = np.unique(protected)
        group_rates = {}
        
        for group in groups:
            mask = protected == group
            if mask.sum() > 0:
                rate = np.mean(y_pred[mask])
                group_rates[group] = float(rate)
        
        rates = list(group_rates.values())
        
        if len(rates) >= 2 and max(rates) > 0:
            di_ratio = min(rates) / max(rates)
        else:
            di_ratio = 1.0
        
        return {
            'disparate_impact_ratio': float(di_ratio),
            'passes_80_percent_rule': di_ratio >= 0.8,
            'group_rates': group_rates
        }
    
    def equalized_odds(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate equalized odds metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected: Protected attribute values
            
        Returns:
            Dictionary with equalized odds metrics
        """
        groups = np.unique(protected)
        tprs = {}
        fprs = {}
        
        for group in groups:
            mask = protected == group
            group_true = y_true[mask]
            group_pred = y_pred[mask]
            
            # TPR (among actual positives)
            pos_mask = group_true == 1
            if pos_mask.sum() > 0:
                tprs[group] = float(np.mean(group_pred[pos_mask]))
            
            # FPR (among actual negatives)
            neg_mask = group_true == 0
            if neg_mask.sum() > 0:
                fprs[group] = float(np.mean(group_pred[neg_mask]))
        
        tpr_values = list(tprs.values())
        fpr_values = list(fprs.values())
        
        tpr_diff = max(tpr_values) - min(tpr_values) if len(tpr_values) >= 2 else 0.0
        fpr_diff = max(fpr_values) - min(fpr_values) if len(fpr_values) >= 2 else 0.0
        
        return {
            'true_positive_rate_difference': float(tpr_diff),
            'false_positive_rate_difference': float(fpr_diff),
            'group_tprs': tprs,
            'group_fprs': fprs
        }
    
    def evaluate(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        protected_attributes: List[str],
        protected_data: Optional[pd.DataFrame] = None,
        decision_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive fairness evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            protected_attributes: List of protected attribute names
            protected_data: DataFrame with protected attribute values
            
        Returns:
            Dictionary with fairness metrics
        """
        logger.info("Running fairness evaluation...")
        
        y_pred = predict_with_threshold(model, X_test, decision_threshold)
        
        # If no protected data, return basic metrics
        if protected_data is None or not protected_attributes:
            return {
                'warning': 'No protected attributes data provided',
                'overall_positive_rate': float(np.mean(y_pred)),
                'recommendations': [
                    'Provide protected attribute data for fairness analysis',
                    'Consider collecting demographic information ethically'
                ]
            }
        
        results = {
            'protected_attributes': protected_attributes,
            'decision_threshold': float(decision_threshold) if decision_threshold is not None else None,
            'metrics_by_attribute': {},
            'overall_fairness': True,
            'recommendations': []
        }
        
        for attr in protected_attributes:
            if attr not in protected_data.columns:
                logger.warning(f"Protected attribute '{attr}' not found in data")
                continue
            
            attr_results = self._evaluate_attribute(
                y_pred, y_test, protected_data[attr]
            )
            results['metrics_by_attribute'][attr] = attr_results
            
            if not attr_results['is_fair']:
                results['overall_fairness'] = False
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        logger.info("Fairness evaluation complete")
        
        return results
    
    def _evaluate_attribute(
        self,
        y_pred: np.ndarray,
        y_test: np.ndarray,
        protected_values: pd.Series
    ) -> Dict[str, Any]:
        """Evaluate fairness for a single protected attribute."""
        groups = protected_values.unique()
        
        group_metrics = {}
        
        for group in groups:
            mask = protected_values == group
            group_size = mask.sum()
            
            if group_size == 0:
                continue
            
            group_pred = y_pred[mask]
            group_true = y_test[mask]
            
            # Positive prediction rate
            positive_rate = np.mean(group_pred)

            # Group-level predictive performance
            accuracy = accuracy_score(group_true, group_pred)
            precision = precision_score(group_true, group_pred, zero_division=0)
            recall = recall_score(group_true, group_pred, zero_division=0)
            f1 = f1_score(group_true, group_pred, zero_division=0)
            
            # True positive rate (if applicable)
            positive_mask = group_true == 1
            if positive_mask.sum() > 0:
                tpr = np.mean(group_pred[positive_mask])
            else:
                tpr = None
            
            # False positive rate
            negative_mask = group_true == 0
            if negative_mask.sum() > 0:
                fpr = np.mean(group_pred[negative_mask])
            else:
                fpr = None
            
            group_metrics[str(group)] = {
                'size': int(group_size),
                'positive_rate': float(positive_rate),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'true_positive_rate': float(tpr) if tpr is not None else None,
                'false_positive_rate': float(fpr) if fpr is not None else None
            }
        
        # Calculate fairness metrics
        positive_rates = [g['positive_rate'] for g in group_metrics.values()]
        accuracies = [g['accuracy'] for g in group_metrics.values()]
        precisions = [g['precision'] for g in group_metrics.values()]
        recalls = [g['recall'] for g in group_metrics.values()]
        f1_scores = [g['f1'] for g in group_metrics.values()]
        
        if len(positive_rates) >= 2:
            # Demographic parity
            dp_ratio = min(positive_rates) / max(positive_rates) if max(positive_rates) > 0 else 1.0
            dp_difference = max(positive_rates) - min(positive_rates)
            
            # Disparate impact (80% rule)
            disparate_impact = dp_ratio
            
            is_fair = disparate_impact >= self.threshold
            accuracy_gap = max(accuracies) - min(accuracies)
            precision_gap = max(precisions) - min(precisions)
            recall_gap = max(recalls) - min(recalls)
            f1_gap = max(f1_scores) - min(f1_scores)
        else:
            dp_ratio = 1.0
            dp_difference = 0.0
            disparate_impact = 1.0
            is_fair = True
            accuracy_gap = 0.0
            precision_gap = 0.0
            recall_gap = 0.0
            f1_gap = 0.0

        worst_group_by_f1 = None
        best_group_by_f1 = None
        if group_metrics:
            sorted_by_f1 = sorted(group_metrics.items(), key=lambda item: item[1]['f1'])
            worst_group_by_f1 = sorted_by_f1[0][0]
            best_group_by_f1 = sorted_by_f1[-1][0]
        
        return {
            'group_metrics': group_metrics,
            'demographic_parity_ratio': float(dp_ratio),
            'demographic_parity_difference': float(dp_difference),
            'disparate_impact': float(disparate_impact),
            'accuracy_gap': float(accuracy_gap),
            'precision_gap': float(precision_gap),
            'recall_gap': float(recall_gap),
            'f1_gap': float(f1_gap),
            'worst_group_by_f1': worst_group_by_f1,
            'best_group_by_f1': best_group_by_f1,
            'is_fair': is_fair,
            'threshold': self.threshold
        }
    
    def _generate_recommendations(
        self,
        results: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on fairness results."""
        recommendations = []
        
        if not results['overall_fairness']:
            recommendations.append(
                "[!] Model shows potential unfairness. Consider the following:"
            )
            
            for attr, metrics in results['metrics_by_attribute'].items():
                if not metrics.get('is_fair', True):
                    di = metrics.get('disparate_impact', 0)
                    recommendations.append(
                        f"  - '{attr}': Disparate impact = {di:.2%} "
                        f"(threshold: {self.threshold:.0%})"
                    )
            
            recommendations.extend([
                "",
                "Suggested mitigations:",
                "1. Review training data for historical bias",
                "2. Apply fairness constraints during training",
                "3. Use post-processing techniques to adjust predictions",
                "4. Consider using fairness-aware algorithms",
                "5. Consult with domain experts and stakeholders"
            ])
        else:
            recommendations.append(
                "[OK] Model meets fairness threshold for all protected attributes."
            )
            recommendations.append(
                "Continue monitoring for fairness as data and model evolve."
            )
        
        return recommendations
    
    def compute_equalized_odds(
        self,
        y_pred: np.ndarray,
        y_test: np.ndarray,
        protected_values: pd.Series
    ) -> Dict[str, Any]:
        """
        Compute equalized odds metric.
        
        Equalized odds requires equal TPR and FPR across groups.
        """
        groups = protected_values.unique()
        
        tprs = []
        fprs = []
        
        for group in groups:
            mask = protected_values == group
            group_pred = y_pred[mask]
            group_true = y_test[mask]
            
            # TPR
            pos_mask = group_true == 1
            if pos_mask.sum() > 0:
                tpr = np.mean(group_pred[pos_mask])
                tprs.append(tpr)
            
            # FPR
            neg_mask = group_true == 0
            if neg_mask.sum() > 0:
                fpr = np.mean(group_pred[neg_mask])
                fprs.append(fpr)
        
        return {
            'tpr_difference': float(max(tprs) - min(tprs)) if len(tprs) >= 2 else 0,
            'fpr_difference': float(max(fprs) - min(fprs)) if len(fprs) >= 2 else 0,
            'equalized_odds_satisfied': (
                (max(tprs) - min(tprs) < 0.1) and (max(fprs) - min(fprs) < 0.1)
                if len(tprs) >= 2 and len(fprs) >= 2 else True
            )
        }
    
    def generate_fairness_report(
        self,
        results: Dict[str, Any],
        format: str = "text"
    ) -> str:
        """Generate formatted fairness report."""
        if format == "markdown":
            return self._generate_markdown_report(results)
        return self._generate_text_report(results)
    
    def _generate_text_report(self, results: Dict[str, Any]) -> str:
        """Generate plain text fairness report."""
        lines = ["=" * 50]
        lines.append("FAIRNESS EVALUATION REPORT")
        lines.append("=" * 50)
        
        lines.append(f"\nOverall Fair: {'Yes' if results.get('overall_fairness', False) else 'No'}")
        lines.append(f"Threshold: {self.threshold:.0%}")
        
        for attr, metrics in results.get('metrics_by_attribute', {}).items():
            lines.append(f"\n--- {attr} ---")
            lines.append(f"Disparate Impact: {metrics.get('disparate_impact', 0):.2%}")
            lines.append(f"Is Fair: {'Yes' if metrics.get('is_fair', False) else 'No'}")
            
            for group, gm in metrics.get('group_metrics', {}).items():
                lines.append(f"  {group}: positive rate = {gm.get('positive_rate', 0):.2%}")
        
        lines.append("\n" + "-" * 50)
        lines.append("RECOMMENDATIONS:")
        for rec in results.get('recommendations', []):
            lines.append(rec)
        
        return "\n".join(lines)
    
    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate Markdown fairness report."""
        lines = ["# Fairness Evaluation Report\n"]
        
        status = "[PASS]" if results.get('overall_fairness', False) else "[ATTENTION]"
        lines.append(f"**Overall Status:** {status}\n")
        
        for attr, metrics in results.get('metrics_by_attribute', {}).items():
            lines.append(f"## {attr}\n")
            lines.append(f"- Disparate Impact: {metrics.get('disparate_impact', 0):.2%}")
            lines.append(f"- Threshold: {self.threshold:.0%}")
            lines.append("")
        
        lines.append("## Recommendations\n")
        for rec in results.get('recommendations', []):
            lines.append(f"- {rec}")
        
        return "\n".join(lines)
