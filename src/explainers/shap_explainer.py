"""
SHAP Explainer - Тайлбарлах үндсэн хөдөлгүүр
==============================================

Машин сургалтын загварын таамаглалуудад SHAP (SHapley Additive exPlanations)
суурилсан тайлбарыг хангана.

SHAP утгууд нь таамаглалд шинж чанар бүрийн хувь нэмрийг илэрхийлж,
локал (ганц бие) болон глобал (ерөнхий) тайлбарыг хангадаг.

Зохиогч: XAI-SHAP Framework
"""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-д суурилсан загвар тайлбарлагч.
    
    Энэ класс дараах боломжуудыг олгоно:
    - Загварын төрлөөс хамаарсан автомат explainer сонголт
    - Ганц таамаглалд зориулсан локал тайлбар
    - Ерөнхий загварын зан үйлд зориулсан глобал тайлбар
    - Feature interaction шинжилгээ
    - Тайлбарын тогтвортой байдлын шинжилгээ
    
    SHAP Explainer-ийн төрлүүд:
    - TreeExplainer: Модны загваруудад (XGBoost, RF, LightGBM)
    - KernelExplainer: Загвар-агностик (удаан боловч аливаа загварт ажилладаг)
    - DeepExplainer: Гүн сургалтын загваруудад
    - LinearExplainer: Шугаман загваруудад
    
    Жишээ:
        >>> explainer = SHAPExplainer(config)
        >>> explanations = explainer.explain(
        ...     model=model,
        ...     X_background=X_train,
        ...     X_explain=X_test,
        ...     feature_names=feature_names
        ... )
        >>> print(explanations['global']['feature_importance'])
    
    Шинж чанарууд:
        config: Тохиргооны менежер instance
        explainer: SHAP explainer обьект
        shap_values: Тооцоолсон SHAP утгууд
        base_value: Хүлээгдэж буй загварын гаралт
    """
    
    def __init__(self, config=None):
        """
        SHAP Explainer эхлүүлэх.
        
        Параметрүүд:
            config: Тохиргооны менежер instance
        """
        self.config = config
        self.explainer = None
        self.shap_values = None
        self.base_value = None
        self._explanation_object = None
        
        # Тохиргооны тохируулгуудыг авах
        if config:
            self.background_samples = config.get("shap.background_samples", 100)
            self.max_display_features = config.get("shap.max_display_features", 20)
            self.explainer_type = config.get("shap.explainer_type", "auto")
        else:
            self.background_samples = 100
            self.max_display_features = 20
            self.explainer_type = "auto"
    
    def explain(
        self,
        model: Any,
        X_background: np.ndarray,
        X_explain: np.ndarray,
        feature_names: Optional[List[str]] = None,
        explanation_type: str = "both",
        sample_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Загварын таамаглалд SHAP тайлбар үүсгэх.
        
        Параметрүүд:
            model: Сургасан машин сургалтын загвар
            X_background: SHAP-д зориулсан арын өгөгдөл (ихэвчлэн сургалтын өгөгдөл)
            X_explain: Тайлбарлах өгөгдөл (ихэвчлэн тест өгөгдөл)
            feature_names: Шинж чанаруудын нэрсийн жагсаалт
            explanation_type: 'global', 'local', эсвэл 'both'
            sample_indices: Локал тайлбарт зориулсан тодорхой жишээнүүд
            
        Буцаах:
            Дараахыг агуулсан dictionary:
            - shap_values: Түүхий SHAP утгууд
            - global: Глобал тайлбарын хэмжигдэхүүнүүд
            - local: Локал тайлбарууд (хэрэв хүссэн бол)
            - explainer: SHAP explainer обьект
        """
        logger.info(f"{explanation_type} SHAP тайлбаруудыг үүсгэж байна")
        
        # Feature нэрсийг хадгалах
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(X_explain.shape[1])]
        
        # Арын өгөгдлийн олонлог үүсгэх (үр ашгийн төлөө дэд түүвэр)
        if len(X_background) > self.background_samples:
            indices = np.random.choice(
                len(X_background),
                self.background_samples,
                replace=False
            )
            background = X_background[indices]
        else:
            background = X_background
        
        # Тохирох explainer үүсгэх
        self.explainer = self._create_explainer(model, background)
        
        # SHAP утгуудыг тооцоолох
        logger.info("SHAP утгуудыг тооцоолж байна...")
        
        if sample_indices is not None:
            X_to_explain = X_explain[sample_indices]
        else:
            X_to_explain = X_explain
        
        self._explanation_object = self.explainer(X_to_explain)
        self.shap_values = self._explanation_object.values
        self.base_value = self._explanation_object.base_values
        
        # Олон ангийн SHAP утгуудыг боловсруулах
        if len(self.shap_values.shape) == 3:
            # Олон ангид эерэг ангийг эсвэл ангиуд дээр дундажийг ашиглах
            self.shap_values = self.shap_values[:, :, 1] if self.shap_values.shape[2] == 2 else self.shap_values.mean(axis=2)
        
        # Үр дүнгийн dictionary бүтээх
        result = {
            'shap_values': self.shap_values,
            'base_value': self.base_value,
            'feature_names': self.feature_names,
            'explainer': self.explainer,
            'explanation_object': self._explanation_object
        }
        
        # Глобал тайлбар нэмэх
        if explanation_type in ['global', 'both']:
            result['global'] = self._compute_global_explanation(
                self.shap_values, X_to_explain
            )
        
        # Локал тайлбарууд нэмэх
        if explanation_type in ['local', 'both']:
            result['local'] = self._compute_local_explanations(
                self.shap_values, X_to_explain, sample_indices
            )
        
        logger.info("SHAP тайлбарууд амжилттай үүсгэгдлээ")
        
        return result
    
    def _create_explainer(self, model: Any, background: np.ndarray) -> shap.Explainer:
        """
        Загварын төрлөөс хамаарч тохирох SHAP explainer үүсгэх.
        
        Параметрүүд:
            model: ML загвар
            background: Арын өгөгдөл
            
        Буцаах:
            SHAP explainer instance
        """
        model_type = type(model).__name__.lower()
        
        # Pipeline бол дотор нь байгаа загварыг авах
        actual_model = model
        if 'pipeline' in model_type:
            # Pipeline-ийн сүүлийн step нь загвар
            actual_model = model.steps[-1][1]
            model_type = type(actual_model).__name__.lower()
            logger.info(f"Pipeline дотор {type(actual_model).__name__} загвар олдлоо")
        
        if self.explainer_type != "auto":
            # Тодорхойлсон explainer төрлийг ашиглах
            return self._create_specific_explainer(
                self.explainer_type, model, background
            )
        
        # Tree-based загварууд
        tree_types = [
            'xgb', 'lgb', 'lightgbm', 'catboost',  # Gradient Boosting
            'randomforest', 'extratrees', 'decisiontree',  # Tree Ensemble
            'gradientboosting', 'adaboost', 'bagging'  # sklearn Ensemble
        ]
        
        if any(tree_type in model_type for tree_type in tree_types):
            logger.info(f"Tree-based загвар ({type(actual_model).__name__}) TreeExplainer ашиглаж байна")
            try:
                return shap.TreeExplainer(actual_model)
            except Exception as e:
                logger.warning(f"TreeExplainer амжилтгүй: {e}. Explainer руу шилжиж байна")
                return shap.Explainer(model, background)
        
        # Neural Network загварууд
        elif 'mlp' in model_type or 'neural' in model_type:
            logger.info("Neural network-д KernelExplainer ашиглаж байна")
            return shap.KernelExplainer(model.predict, background)
        
        # Шугаман загварууд (Logistic Regression, Ridge, ElasticNet)
        elif any(linear_type in model_type for linear_type in 
                 ['linear', 'logistic', 'ridge', 'lasso', 'elasticnet']):
            logger.info(f"Шугаман загвар ({type(actual_model).__name__}) LinearExplainer ашиглаж байна")
            try:
                # Pipeline бол scaler-тай background өгөх
                if 'pipeline' in type(model).__name__.lower():
                    # Scaler-аар хувиргах
                    background_scaled = model.named_steps.get('scaler', lambda x: x)
                    if hasattr(background_scaled, 'transform'):
                        bg_transformed = background_scaled.transform(background)
                        return shap.LinearExplainer(actual_model, bg_transformed)
                return shap.LinearExplainer(actual_model, background)
            except Exception as e:
                logger.warning(f"LinearExplainer амжилтгүй: {e}. KernelExplainer руу шилжиж байна")
                return shap.KernelExplainer(model.predict, background)
        
        # SVM загварууд
        elif any(svm_type in model_type for svm_type in ['svc', 'svr', 'svm']):
            logger.info(f"SVM загвар ({type(actual_model).__name__}) KernelExplainer ашиглаж байна")
            # SVM-д KernelExplainer ашиглах (удаан боловч найдвартай)
            if hasattr(model, 'predict_proba'):
                return shap.KernelExplainer(model.predict_proba, background)
            else:
                return shap.KernelExplainer(model.predict, background)
        
        else:
            # Автоматаар хамгийн сайн аргыг сонгох Explainer-г ашиглах
            logger.info(f"Автоматаар илрүүлсэн Explainer ашиглаж байна ({type(actual_model).__name__})")
            return shap.Explainer(model, background)
    
    def _create_specific_explainer(
        self,
        explainer_type: str,
        model: Any,
        background: np.ndarray
    ) -> shap.Explainer:
        """Тодорхой explainer төрөл үүсгэх."""
        if explainer_type == "tree":
            return shap.TreeExplainer(model)
        elif explainer_type == "kernel":
            return shap.KernelExplainer(model.predict, background)
        elif explainer_type == "deep":
            return shap.DeepExplainer(model, background)
        elif explainer_type == "linear":
            return shap.LinearExplainer(model, background)
        else:
            return shap.Explainer(model, background)
    
    def _compute_global_explanation(
        self,
        shap_values: np.ndarray,
        X: np.ndarray
    ) -> Dict[str, Any]:
        """
        Глобал тайлбарын хэмжигдэхүүнүүдийг тооцоолох.
        
        Глобал тайлбар нь бүх таамаглалд ерөнхий
        feature importance-ийг харуулдаг.
        """
        # Feature бүрийн дундаж абсолют SHAP утга
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Feature importance-ийн эрэмбэ
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        # Top N features
        top_features = importance_df.head(self.max_display_features)
        
        return {
            'feature_importance': importance_df.to_dict(orient='records'),
            'top_features': top_features['feature'].tolist(),
            'mean_abs_shap': dict(zip(self.feature_names, mean_abs_shap)),
            'summary': self._generate_global_summary(importance_df)
        }
    
    def _compute_local_explanations(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        sample_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Compute local explanations for individual predictions.
        
        Local explanations show how each feature contributes
        to a specific prediction.
        """
        local_explanations = []
        
        # Explain first N samples if indices not specified
        n_samples = min(10, len(shap_values))
        indices = sample_indices if sample_indices else list(range(n_samples))
        
        for i, idx in enumerate(indices):
            if i >= len(shap_values):
                break
                
            sample_shap = shap_values[i]
            sample_X = X[i] if i < len(X) else None
            
            # Sort features by absolute SHAP value
            sorted_indices = np.argsort(np.abs(sample_shap))[::-1]
            
            # Handle base_value which could be scalar, 1D or 2D array
            if hasattr(self.base_value, '__len__'):
                if len(self.base_value.shape) > 1:
                    base_val = float(self.base_value[i, 0]) if self.base_value.shape[1] > 0 else float(self.base_value[i].mean())
                else:
                    base_val = float(self.base_value[i]) if i < len(self.base_value) else float(self.base_value.mean())
            else:
                base_val = float(self.base_value)
            
            explanation = {
                'sample_index': idx,
                'base_value': base_val,
                'contributions': [
                    {
                        'feature': self.feature_names[j],
                        'shap_value': float(sample_shap[j]),
                        'feature_value': float(sample_X[j]) if sample_X is not None else None,
                        'impact': 'positive' if sample_shap[j] > 0 else 'negative'
                    }
                    for j in sorted_indices[:self.max_display_features]
                ],
                'top_positive': self._get_top_contributors(sample_shap, positive=True),
                'top_negative': self._get_top_contributors(sample_shap, positive=False)
            }
            
            local_explanations.append(explanation)
        
        return {
            'explanations': local_explanations,
            'n_samples': len(local_explanations)
        }
    
    def _get_top_contributors(
        self,
        shap_values: np.ndarray,
        positive: bool = True,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Get top positive or negative contributors."""
        if positive:
            indices = np.argsort(shap_values)[::-1][:top_n]
            contributors = [
                {'feature': self.feature_names[i], 'shap_value': float(shap_values[i])}
                for i in indices if shap_values[i] > 0
            ]
        else:
            indices = np.argsort(shap_values)[:top_n]
            contributors = [
                {'feature': self.feature_names[i], 'shap_value': float(shap_values[i])}
                for i in indices if shap_values[i] < 0
            ]
        
        return contributors
    
    def _generate_global_summary(self, importance_df: pd.DataFrame) -> str:
        """Generate human-readable summary of global explanations."""
        top_3 = importance_df.head(3)['feature'].tolist()
        
        return (
            f"The top 3 most important features are: {', '.join(top_3)}. "
            f"These features have the highest mean absolute SHAP values, "
            f"indicating they contribute most to model predictions on average."
        )
    
    def get_interaction_effects(
        self,
        feature1: str,
        feature2: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute SHAP interaction values between features.
        
        Args:
            feature1: First feature name
            feature2: Second feature name (optional)
            
        Returns:
            Interaction values array
        """
        if not hasattr(self.explainer, 'shap_interaction_values'):
            logger.warning("Interaction values not supported for this explainer type")
            return None
        
        # Find feature indices
        idx1 = self.feature_names.index(feature1)
        
        if feature2:
            idx2 = self.feature_names.index(feature2)
            return self.explainer.shap_interaction_values[:, idx1, idx2]
        
        return self.explainer.shap_interaction_values[:, idx1, :]
    
    def explain_single_prediction(
        self,
        X_single: np.ndarray,
        return_natural_language: bool = False
    ) -> Dict[str, Any]:
        """
        Explain a single prediction in detail.
        
        Args:
            X_single: Single sample to explain (1D or 2D array)
            return_natural_language: Return human-readable explanation
            
        Returns:
            Detailed explanation dictionary
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call explain() first.")
        
        # Ensure 2D input
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)
        
        # Compute SHAP values
        explanation = self.explainer(X_single)
        shap_vals = explanation.values[0]
        base = explanation.base_values[0] if hasattr(explanation.base_values, '__len__') else explanation.base_values
        
        result = {
            'base_value': float(base),
            'predicted_value': float(base + shap_vals.sum()),
            'shap_values': dict(zip(self.feature_names, shap_vals.tolist())),
            'contributions': sorted(
                [{'feature': f, 'contribution': float(v)} 
                 for f, v in zip(self.feature_names, shap_vals)],
                key=lambda x: abs(x['contribution']),
                reverse=True
            )
        }
        
        if return_natural_language:
            result['explanation_text'] = self._generate_natural_language_explanation(
                shap_vals, base, X_single[0]
            )
        
        return result
    
    def _generate_natural_language_explanation(
        self,
        shap_values: np.ndarray,
        base_value: float,
        feature_values: np.ndarray
    ) -> str:
        """Generate human-readable explanation text."""
        sorted_indices = np.argsort(np.abs(shap_values))[::-1][:5]
        
        explanations = []
        for idx in sorted_indices:
            feature = self.feature_names[idx]
            shap_val = shap_values[idx]
            feat_val = feature_values[idx]
            direction = "increases" if shap_val > 0 else "decreases"
            
            explanations.append(
                f"'{feature}' (value: {feat_val:.2f}) {direction} the prediction by {abs(shap_val):.3f}"
            )
        
        text = f"Starting from a baseline prediction of {base_value:.3f}:\n"
        text += "\n".join(f"- {exp}" for exp in explanations)
        text += f"\n\nFinal prediction: {base_value + shap_values.sum():.3f}"
        
        return text
