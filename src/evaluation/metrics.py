"""
Model Evaluator - Загварын иж бүрэн үнэлгээ
=============================================

Загварын гүйцэтгэл болон тайлбарын чанарын
үнэлгээний хэмжигдэхүүнүүдийг хангана.

Зохиогч: XAI-SHAP Framework
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from src.utils.helpers import get_binary_probability_scores, predict_with_threshold

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Олон хэмжигдэхүүнтэй загварын иж бүрэн үнэлгээ.
    
    Үнэлдэг:
    - Classification хэмжигдэхүүнүүд (accuracy, precision, recall, F1, AUC)
    - Regression хэмжигдэхүүнүүд (MSE, RMSE, MAE, R²)
    - Тайлбарын чанарын хэмжигдэхүүнүүд
    - Таамаглалын тогтвортой байдал
    
    Жишээ:
        >>> evaluator = ModelEvaluator(config)
        >>> results = evaluator.evaluate(model, X_test, y_test)
        >>> print(results['classification']['accuracy'])
    """
    
    def __init__(self, config=None):
        """
        ModelEvaluator эхлүүлэх.
        
        Параметрүүд:
            config: Тохиргооны менежер instance
        """
        self.config = config
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Classification загварын гүйцэтгэлийг үнэлэх.
        
        Параметрүүд:
            y_true: Бодит шошго
            y_pred: Таамагласан шошго
            y_proba: Таамагласан магадлалууд (заавал биш)
            
        Буцаах:
            Classification хэмжигдэхүүнүүдтэй dictionary
        """
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }
        
        # Binary classification-д AUC
        if len(np.unique(y_true)) == 2 and y_proba is not None:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
            except Exception as e:
                logger.warning(f"AUC тооцоолж чадсангүй: {e}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Regression загварын гүйцэтгэлийг үнэлэх.
        
        Параметрүүд:
            y_true: Бодит утгууд
            y_pred: Таамагласан утгууд
            
        Буцаах:
            Regression хэмжигдэхүүнүүдтэй dictionary
        """
        mse = mean_squared_error(y_true, y_pred)
        
        return {
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred))
        }
    
    def evaluate(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        shap_values: Optional[np.ndarray] = None,
        task_type: Optional[str] = None,
        decision_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Загварын иж бүрэн үнэлгээ.
        
        Параметрүүд:
            model: Сургасан загвар
            X_test: Тест шинж чанарууд
            y_test: Тест шошго
            shap_values: SHAP утгууд (заавал биш)
            task_type: 'classification' эсвэл 'regression' (автомат илрүүлэх)
            
        Буцаах:
            Үнэлгээний хэмжигдэхүүнүүдтэй dictionary
        """
        logger.info("Загварын үнэлгээг ажиллуулж байна...")
        
        # Даалгаврын төрлийг автомат илрүүлэх
        if task_type is None:
            unique_values = np.unique(y_test)
            task_type = 'classification' if len(unique_values) <= 20 else 'regression'
        
        results = {
            'даалгаврын_төрөл': task_type
        }
        
        # Таамаглалуудыг авах
        if task_type == 'classification':
            y_pred = predict_with_threshold(model, X_test, decision_threshold)
            y_proba = get_binary_probability_scores(model, X_test)
            if decision_threshold is not None and y_proba is not None:
                results['шийдвэрийн_босго'] = float(decision_threshold)
        else:
            y_pred = model.predict(X_test)
            y_proba = None
        
        # Даалгаврын төрлөөс хамаарсан хэмжигдэхүүнүүд
        if task_type == 'classification':
            results['classification'] = self._evaluate_classification(y_test, y_pred, y_proba)
        else:
            results['regression'] = self._evaluate_regression(y_test, y_pred)
        
        # Тайлбарын чанарын хэмжигдэхүүнүүд
        if shap_values is not None:
            results['тайлбарын_чанар'] = self._evaluate_explanations(shap_values)
        
        # Таамаглалын хуваарилалт
        results['таамаглалын_хуваарилалт'] = self._analyze_predictions(y_pred, task_type)
        
        logger.info("Үнэлгээ дууслаа")
        
        return results
    
    def _evaluate_classification(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Classification загварыг үнэлэх."""
        metrics = {
            'нарийвчлал': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_оноо': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        }
        
        # Binary classification-д AUC
        if len(np.unique(y_test)) == 2 and y_proba is not None:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba))
            except Exception as e:
                logger.warning(f"AUC тооцоолж чадсангүй: {e}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics['дэлгэрэнгүй_тайлан'] = report
        
        return metrics
    
    def _evaluate_regression(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Regression загварыг үнэлэх."""
        mse = mean_squared_error(y_test, y_pred)
        
        return {
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'r2_оноо': float(r2_score(y_test, y_pred))
        }
    
    def _evaluate_explanations(
        self,
        shap_values: np.ndarray
    ) -> Dict[str, Any]:
        """Тайлбарын чанарыг үнэлэх."""
        # Тогтвортой байдал: Дээжүүдийн шинж чанарын ач холбогдлын стандарт хазайлт
        feature_std = np.std(np.abs(shap_values), axis=0)
        
        # Хамрах хүрээ: Хэдэн шинж чанар тэг биш ач холбогдолтой вэ
        feature_coverage = np.mean(np.abs(shap_values) > 1e-6, axis=0)
        
        # Сийрэг байдал: Дээж тус бүрийн чухал шинж чанаруудын дундаж тоо
        significant_threshold = np.abs(shap_values).max() * 0.01
        sparsity = np.mean(np.sum(np.abs(shap_values) > significant_threshold, axis=1))
        
        return {
            'шинж_чанарын_ач_холбогдлын_тогтвортой_байдал': float(np.mean(feature_std)),
            'шинж_чанарын_хамрах_хүрээ': float(np.mean(feature_coverage)),
            'дундаж_чухал_шинж_чанарууд': float(sparsity),
            'нийт_атрибуци': float(np.abs(shap_values).sum(axis=1).mean())
        }
    
    def _analyze_predictions(
        self,
        y_pred: np.ndarray,
        task_type: str
    ) -> Dict[str, Any]:
        """Таамаглалын хуваарилалтыг шинжлэх."""
        if task_type == 'classification':
            unique, counts = np.unique(y_pred, return_counts=True)
            distribution = dict(zip(unique.astype(str).tolist(), counts.tolist()))
            
            return {
                'өвөрмөц_таамаглалууд': len(unique),
                'хуваарилалт': distribution
            }
        else:
            return {
                'дундаж': float(np.mean(y_pred)),
                'стандарт_хазайлт': float(np.std(y_pred)),
                'хамгийн_бага': float(np.min(y_pred)),
                'хамгийн_их': float(np.max(y_pred)),
                'медиан': float(np.median(y_pred))
            }
    
    def generate_report(
        self,
        results: Dict[str, Any],
        format: str = "text"
    ) -> str:
        """
        Үнэлгээний тайлан үүсгэх.
        
        Параметрүүд:
            results: Үнэлгээний үр дүнгийн dictionary
            format: Гаралтын формат ('text', 'markdown', 'html')
            
        Буцаах:
            Форматлагдсан тайлангийн string
        """
        if format == "markdown":
            return self._generate_markdown_report(results)
        elif format == "html":
            return self._generate_html_report(results)
        else:
            return self._generate_text_report(results)
    
    def _generate_text_report(self, results: Dict[str, Any]) -> str:
        """Энгийн текст тайлан үүсгэх."""
        lines = ["=" * 50]
        lines.append("ЗАГВАРЫН ҮНЭЛГЭЭНИЙ ТАЙЛАН")
        lines.append("=" * 50)
        lines.append(f"\nДаалгаврын төрөл: {results.get('даалгаврын_төрөл', results.get('task_type', 'N/A'))}")
        
        if 'classification' in results:
            metrics = results['classification']
            lines.append("\n--- Classification Хэмжигдэхүүнүүд ---")
            lines.append(f"Нарийвчлал:  {metrics.get('нарийвчлал', metrics.get('accuracy', 0)):.4f}")
            lines.append(f"Precision:   {metrics.get('precision', 0):.4f}")
            lines.append(f"Recall:      {metrics.get('recall', 0):.4f}")
            lines.append(f"F1 Оноо:     {metrics.get('f1_оноо', metrics.get('f1_score', 0)):.4f}")
            if 'roc_auc' in metrics:
                lines.append(f"ROC AUC:     {metrics['roc_auc']:.4f}")
        
        if 'regression' in results:
            metrics = results['regression']
            lines.append("\n--- Regression Хэмжигдэхүүнүүд ---")
            lines.append(f"MSE:       {metrics['mse']:.4f}")
            lines.append(f"RMSE:      {metrics['rmse']:.4f}")
            lines.append(f"MAE:       {metrics['mae']:.4f}")
            lines.append(f"R² Оноо:   {metrics.get('r2_оноо', metrics.get('r2_score', 0)):.4f}")
        
        if 'тайлбарын_чанар' in results or 'explanation_quality' in results:
            eq = results.get('тайлбарын_чанар', results.get('explanation_quality', {}))
            lines.append("\n--- Тайлбарын Чанар ---")
            lines.append(f"Тогтвортой байдал: {eq.get('шинж_чанарын_ач_холбогдлын_тогтвортой_байдал', eq.get('feature_importance_consistency', 0)):.4f}")
            lines.append(f"Хамрах хүрээ:      {eq.get('шинж_чанарын_хамрах_хүрээ', eq.get('feature_coverage', 0)):.4f}")
            lines.append(f"Дундаж чухал шинж чанарууд: {eq.get('дундаж_чухал_шинж_чанарууд', eq.get('average_significant_features', 0)):.1f}")
        
        lines.append("\n" + "=" * 50)
        
        return "\n".join(lines)
    
    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Markdown тайлан үүсгэх."""
        lines = ["# Загварын Үнэлгээний Тайлан\n"]
        lines.append(f"**Даалгаврын төрөл:** {results.get('даалгаврын_төрөл', results.get('task_type', 'N/A'))}\n")
        
        if 'classification' in results:
            metrics = results['classification']
            lines.append("## Classification Хэмжигдэхүүнүүд\n")
            lines.append("| Хэмжигдэхүүн | Утга |")
            lines.append("|--------------|------|")
            lines.append(f"| Нарийвчлал | {metrics.get('нарийвчлал', metrics.get('accuracy', 0)):.4f} |")
            lines.append(f"| Precision | {metrics.get('precision', 0):.4f} |")
            lines.append(f"| Recall | {metrics.get('recall', 0):.4f} |")
            lines.append(f"| F1 Оноо | {metrics.get('f1_оноо', metrics.get('f1_score', 0)):.4f} |")
            if 'roc_auc' in metrics:
                lines.append(f"| ROC AUC | {metrics['roc_auc']:.4f} |")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """HTML тайлан үүсгэх."""
        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                table { border-collapse: collapse; width: 100%; max-width: 600px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Загварын Үнэлгээний Тайлан</h1>
        """
        
        if 'classification' in results:
            metrics = results['classification']
            html += """
            <h2>Classification Хэмжигдэхүүнүүд</h2>
            <table>
                <tr><th>Хэмжигдэхүүн</th><th>Утга</th></tr>
            """
            metric_names = {
                'accuracy': 'Нарийвчлал',
                'нарийвчлал': 'Нарийвчлал',
                'precision': 'Precision',
                'recall': 'Recall',
                'f1_score': 'F1 Оноо',
                'f1_оноо': 'F1 Оноо',
                'roc_auc': 'ROC AUC'
            }
            for key in ['нарийвчлал', 'accuracy', 'precision', 'recall', 'f1_оноо', 'f1_score', 'roc_auc']:
                if key in metrics:
                    display_name = metric_names.get(key, key.replace('_', ' ').title())
                    html += f"<tr><td>{display_name}</td><td>{metrics[key]:.4f}</td></tr>"
            html += "</table>"
        
        html += "</body></html>"
        
        return html
