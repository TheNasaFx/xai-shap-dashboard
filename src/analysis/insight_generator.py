"""
Insight Generator - Автомат Дүгнэлт Үүсгэгч
============================================

SHAP утгууд болон загварын өгөгдлөөс автоматаар
хүнд ойлгомжтой дүгнэлт, санал болгох зүйлс үүсгэнэ.

График харуулахаас гадна "Энэ юу гэж байна?" 
гэдгийг тайлбарлаж өгнө.

Зохиогч: XAI-SHAP Framework
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class InsightGenerator:
    """
    SHAP тайлбараас автомат дүгнэлт үүсгэх класс.
    
    Энэ класс дараах боломжуудыг олгоно:
    - Шинж чанарын ач холбогдлын дүгнэлт
    - Threshold/bosgo утга илрүүлэх
    - Шинж чанаруудын харилцан үйлчлэлийн дүгнэлт
    - Эрсдэлийн хүчин зүйлс тодорхойлох
    - Хэрэглэгчид ойлгомжтой тайлбар үүсгэх
    
    Жишээ:
        >>> generator = InsightGenerator()
        >>> insights = generator.generate(
        ...     shap_values=shap_values,
        ...     X=X_test,
        ...     feature_names=feature_names,
        ...     predictions=y_pred
        ... )
        >>> print(insights['summary'])
    """
    
    def __init__(self, config=None):
        """
        InsightGenerator эхлүүлэх.
        
        Параметрүүд:
            config: Тохиргооны менежер instance
        """
        self.config = config
        self.importance_threshold = 0.05  # 5% доош бол бага ач холбогдолтой
        self.correlation_threshold = 0.3  # Feature-SHAP correlation threshold
        
    def generate(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        predictions: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None,
        model_type: str = "classification"
    ) -> Dict[str, Any]:
        """
        SHAP утгуудаас иж бүрэн дүгнэлт үүсгэх.
        
        Параметрүүд:
            shap_values: SHAP утгуудын массив [n_samples, n_features]
            X: Feature утгуудын массив
            feature_names: Feature нэрсийн жагсаалт
            predictions: Таамаглалууд (заавал биш)
            y_true: Бодит утгууд (заавал биш)
            model_type: 'classification' эсвэл 'regression'
            
        Буцаах:
            Дүгнэлтүүдтэй dictionary
        """
        logger.info("Автомат дүгнэлт үүсгэж байна...")
        
        insights = {
            'feature_importance': self._analyze_feature_importance(shap_values, feature_names),
            'thresholds': self._detect_thresholds(shap_values, X, feature_names),
            'interactions': self._analyze_interactions(shap_values, X, feature_names),
            'risk_factors': self._identify_risk_factors(shap_values, X, feature_names, model_type),
            'protective_factors': self._identify_protective_factors(shap_values, X, feature_names, model_type),
            'feature_effects': self._analyze_feature_effects(shap_values, X, feature_names),
            'summary': None,
            'key_findings': []
        }
        
        # Predictions-тэй холбоотой дүгнэлт
        if predictions is not None:
            insights['prediction_insights'] = self._analyze_predictions(
                shap_values, predictions, feature_names
            )
        
        # Гол олдворууд нэгтгэх
        insights['key_findings'] = self._compile_key_findings(insights)
        insights['summary'] = self._generate_summary_text(insights)
        
        return insights
    
    def _analyze_feature_importance(
        self,
        shap_values: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Шинж чанарын ач холбогдлын шинжилгээ."""
        
        # Дундаж абсолют SHAP утга
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        total_importance = mean_abs_shap.sum()
        
        # Эрэмбэлэх
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        
        importance_list = []
        cumulative = 0
        
        for idx in sorted_idx:
            importance = mean_abs_shap[idx]
            relative_importance = importance / total_importance * 100
            cumulative += relative_importance
            
            importance_list.append({
                'feature': feature_names[idx],
                'importance': float(importance),
                'relative_percent': float(relative_importance),
                'cumulative_percent': float(cumulative),
                'rank': len(importance_list) + 1
            })
        
        # Тодорхойлогч шинж чанарууд (80% нөлөөг тайлбарлах)
        dominant_features = [f for f in importance_list if f['cumulative_percent'] <= 80]
        
        return {
            'ranked_features': importance_list,
            'total_features': len(feature_names),
            'dominant_features': dominant_features,
            'dominant_count': len(dominant_features),
            'top_feature': importance_list[0] if importance_list else None,
            'insight': self._importance_insight(importance_list)
        }
    
    def _importance_insight(self, importance_list: List[Dict]) -> str:
        """Ач холбогдлын дүгнэлт текст."""
        if not importance_list:
            return "Ач холбогдлын мэдээлэл байхгүй"
        
        top = importance_list[0]
        dominant = [f for f in importance_list if f['cumulative_percent'] <= 80]
        
        insight = f"'{top['feature']}' нь таамаглалын {top['relative_percent']:.1f}%-ийг тайлбарласан хамгийн чухал шинж чанар."
        
        if len(dominant) > 1:
            insight += f" Дээд {len(dominant)} шинж чанар нийт нөлөөний 80%-ийг тооцсон."
        
        return insight
    
    def _detect_thresholds(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Threshold/босго утгуудыг илрүүлэх."""
        thresholds = {}
        
        for i, feature in enumerate(feature_names):
            feature_vals = X[:, i]
            shap_vals = shap_values[:, i]
            
            # SHAP утга эерэг/сөрөг болох цэгийг олох
            threshold_info = self._find_threshold(feature_vals, shap_vals)
            
            if threshold_info is not None:
                thresholds[feature] = threshold_info
        
        return {
            'detected_thresholds': thresholds,
            'count': len(thresholds)
        }
    
    def _find_threshold(
        self,
        feature_vals: np.ndarray,
        shap_vals: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Нэг feature-т threshold олох."""
        # NaN-гүй өгөгдөл (object dtype-д pd.isnull ашиглах)
        try:
            feature_vals = np.asarray(feature_vals, dtype=np.float64)
            shap_vals = np.asarray(shap_vals, dtype=np.float64)
            mask = ~(np.isnan(feature_vals) | np.isnan(shap_vals))
        except (ValueError, TypeError):
            mask = ~(pd.isnull(feature_vals) | pd.isnull(shap_vals))
        if mask.sum() < 20:
            return None
        
        feat = feature_vals[mask]
        shap = shap_vals[mask]
        
        # Feature утгаар эрэмбэлэх
        sorted_idx = np.argsort(feat)
        feat_sorted = feat[sorted_idx]
        shap_sorted = shap[sorted_idx]
        
        # SHAP тэмдэг өөрчлөгдөх цэгийг олох
        sign_changes = np.where(np.diff(np.sign(shap_sorted)) != 0)[0]
        
        if len(sign_changes) == 0:
            return None
        
        # Хамгийн тод өөрчлөлтийг олох
        main_change_idx = sign_changes[len(sign_changes) // 2]
        threshold_value = (feat_sorted[main_change_idx] + feat_sorted[main_change_idx + 1]) / 2
        
        # Корреляци
        corr, _ = stats.spearmanr(feat, shap)
        
        return {
            'threshold_value': float(threshold_value),
            'direction': 'positive' if corr > 0 else 'negative',
            'correlation': float(corr),
            'confidence': min(1.0, abs(corr) * 1.5),
            'description': self._threshold_description(threshold_value, corr)
        }
    
    def _threshold_description(self, threshold: float, corr: float) -> str:
        """Threshold-ийн тайлбар текст."""
        direction = "нэмэгдүүлдэг" if corr > 0 else "бууруулдаг"
        return f"~{threshold:.2f}-аас дээш үед таамаглалыг {direction}"
    
    def _analyze_interactions(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        top_n: int = 5
    ) -> Dict[str, Any]:
        """Шинж чанаруудын харилцан үйлчлэлийн шинжилгээ."""
        n_features = len(feature_names)
        interactions = []
        
        # Хос feature-үүдийн SHAP correlation тооцох
        for i in range(min(n_features, 10)):  # Дээд 10 feature
            for j in range(i + 1, min(n_features, 10)):
                # SHAP утгуудын үржвэрийн дисперс
                interaction_strength = np.abs(shap_values[:, i] * shap_values[:, j]).mean()
                
                if interaction_strength > 0.01:
                    interactions.append({
                        'feature_1': feature_names[i],
                        'feature_2': feature_names[j],
                        'interaction_strength': float(interaction_strength),
                        'direction': self._interaction_direction(
                            shap_values[:, i], 
                            shap_values[:, j]
                        )
                    })
        
        # Эрэмбэлэх
        interactions = sorted(
            interactions, 
            key=lambda x: x['interaction_strength'], 
            reverse=True
        )[:top_n]
        
        return {
            'top_interactions': interactions,
            'strongest_interaction': interactions[0] if interactions else None,
            'insight': self._interaction_insight(interactions)
        }
    
    def _interaction_direction(self, shap1: np.ndarray, shap2: np.ndarray) -> str:
        """Харилцан үйлчлэлийн чиглэл."""
        corr = np.corrcoef(shap1, shap2)[0, 1]
        if corr > 0.2:
            return "synergistic (хамтдаа бэхжүүлдэг)"
        elif corr < -0.2:
            return "antagonistic (эсрэгцдэг)"
        else:
            return "independent (бие даасан)"
    
    def _interaction_insight(self, interactions: List[Dict]) -> str:
        """Харилцан үйлчлэлийн дүгнэлт."""
        if not interactions:
            return "Тод харилцан үйлчлэл илрээгүй"
        
        top = interactions[0]
        return f"'{top['feature_1']}' ба '{top['feature_2']}' хамгийн их харилцан үйлчлэлтэй ({top['direction']})"
    
    def _identify_risk_factors(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        model_type: str
    ) -> List[Dict[str, Any]]:
        """Эрсдэлийн хүчин зүйлс (эерэг нөлөөтэй)."""
        risk_factors = []
        
        for i, feature in enumerate(feature_names):
            # Эерэг SHAP утгуудын дундаж
            positive_shap_mean = shap_values[:, i][shap_values[:, i] > 0].mean() if (shap_values[:, i] > 0).any() else 0
            positive_count = (shap_values[:, i] > 0).sum()
            positive_ratio = positive_count / len(shap_values)
            
            if positive_shap_mean > 0.05 and positive_ratio > 0.3:
                # Аль утгуудад эерэг болж байгаа
                positive_mask = shap_values[:, i] > 0
                typical_value = np.median(X[positive_mask, i])
                
                risk_factors.append({
                    'feature': feature,
                    'avg_positive_impact': float(positive_shap_mean),
                    'frequency': float(positive_ratio),
                    'typical_risky_value': float(typical_value),
                    'description': f"'{feature}' ~{typical_value:.2f} орчим үед эрсдэл нэмэгддэг"
                })
        
        return sorted(risk_factors, key=lambda x: x['avg_positive_impact'], reverse=True)[:5]
    
    def _identify_protective_factors(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        model_type: str
    ) -> List[Dict[str, Any]]:
        """Хамгаалах хүчин зүйлс (сөрөг нөлөөтэй)."""
        protective_factors = []
        
        for i, feature in enumerate(feature_names):
            # Сөрөг SHAP утгуудын дундаж
            negative_shap_mean = shap_values[:, i][shap_values[:, i] < 0].mean() if (shap_values[:, i] < 0).any() else 0
            negative_count = (shap_values[:, i] < 0).sum()
            negative_ratio = negative_count / len(shap_values)
            
            if negative_shap_mean < -0.05 and negative_ratio > 0.3:
                negative_mask = shap_values[:, i] < 0
                typical_value = np.median(X[negative_mask, i])
                
                protective_factors.append({
                    'feature': feature,
                    'avg_negative_impact': float(negative_shap_mean),
                    'frequency': float(negative_ratio),
                    'typical_protective_value': float(typical_value),
                    'description': f"'{feature}' ~{typical_value:.2f} орчим үед эрсдэл буурдаг"
                })
        
        return sorted(protective_factors, key=lambda x: x['avg_negative_impact'])[:5]
    
    def _analyze_feature_effects(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Feature бүрийн нөлөөний шинжилгээ."""
        effects = {}
        
        for i, feature in enumerate(feature_names[:15]):  # Дээд 15 feature
            shap = shap_values[:, i]
            feat = X[:, i]
            
            # Корреляци (object dtype-д pd.isnull ашиглах)
            try:
                feat = np.asarray(feat, dtype=np.float64)
                shap = np.asarray(shap, dtype=np.float64)
                valid_mask = ~(np.isnan(feat) | np.isnan(shap))
            except (ValueError, TypeError):
                valid_mask = ~(pd.isnull(feat) | pd.isnull(shap))
            if valid_mask.sum() < 10:
                continue
            
            corr, p_value = stats.spearmanr(feat[valid_mask], shap[valid_mask])
            
            # Хамаарлын төрөл
            if abs(corr) > 0.5:
                relationship = "strong_linear"
            elif abs(corr) > 0.2:
                relationship = "moderate_linear"
            else:
                relationship = "non_linear_or_weak"
            
            effects[feature] = {
                'mean_shap': float(shap.mean()),
                'std_shap': float(shap.std()),
                'min_shap': float(shap.min()),
                'max_shap': float(shap.max()),
                'correlation_with_value': float(corr),
                'p_value': float(p_value),
                'relationship_type': relationship,
                'direction': 'positive' if corr > 0 else 'negative' if corr < 0 else 'neutral'
            }
        
        return effects
    
    def _analyze_predictions(
        self,
        shap_values: np.ndarray,
        predictions: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Таамаглалтай холбоотой дүгнэлт."""
        # Эерэг таамаглалуудын шинжилгээ
        if len(np.unique(predictions)) == 2:  # Binary classification
            positive_mask = predictions == 1
            negative_mask = predictions == 0
            
            # Эерэг/сөрөг бүлгүүдэд dominant feature
            pos_importance = np.abs(shap_values[positive_mask]).mean(axis=0) if positive_mask.any() else np.zeros(len(feature_names))
            neg_importance = np.abs(shap_values[negative_mask]).mean(axis=0) if negative_mask.any() else np.zeros(len(feature_names))
            
            return {
                'positive_class_drivers': [
                    {'feature': feature_names[i], 'importance': float(pos_importance[i])}
                    for i in np.argsort(pos_importance)[::-1][:5]
                ],
                'negative_class_drivers': [
                    {'feature': feature_names[i], 'importance': float(neg_importance[i])}
                    for i in np.argsort(neg_importance)[::-1][:5]
                ],
                'class_distribution': {
                    'positive': int(positive_mask.sum()),
                    'negative': int(negative_mask.sum())
                }
            }
        
        return {}
    
    def _compile_key_findings(self, insights: Dict[str, Any]) -> List[str]:
        """Гол олдворуудыг нэгтгэх."""
        findings = []
        
        # Feature importance
        if insights['feature_importance']['top_feature']:
            top = insights['feature_importance']['top_feature']
            findings.append(
                f"[KEY] '{top['feature']}' нь хамгийн чухал шинж чанар ({top['relative_percent']:.1f}%)"
            )
        
        # Dominant features
        dom_count = insights['feature_importance']['dominant_count']
        total = insights['feature_importance']['total_features']
        findings.append(
            f"[DATA] {total} шинж чанараас {dom_count} нь таамаглалын 80%-ийг тайлбарладаг"
        )
        
        # Thresholds
        if insights['thresholds']['count'] > 0:
            findings.append(
                f"[TARGET] {insights['thresholds']['count']} шинж чанарт threshold илэрсэн"
            )
        
        # Interactions
        if insights['interactions']['strongest_interaction']:
            inter = insights['interactions']['strongest_interaction']
            findings.append(
                f"[LINK] '{inter['feature_1']}' ба '{inter['feature_2']}' хамгийн их харилцан үйлчлэлтэй"
            )
        
        # Risk factors
        if insights['risk_factors']:
            risk = insights['risk_factors'][0]
            findings.append(
                f"[WARN] Эрсдэлийн гол хүчин зүйл: {risk['feature']}"
            )
        
        # Protective factors
        if insights['protective_factors']:
            prot = insights['protective_factors'][0]
            findings.append(
                f"[SAFE] Хамгаалах гол хүчин зүйл: {prot['feature']}"
            )
        
        return findings
    
    def _generate_summary_text(self, insights: Dict[str, Any]) -> str:
        """Хэрэглэгчид ойлгомжтой хураангуй текст."""
        lines = [
            "=" * 60,
            "            АВТОМАТ ДҮГНЭЛТИЙН ТАЙЛАН",
            "=" * 60,
            ""
        ]
        
        # Key findings
        lines.append("[FINDINGS] ГОЛ ОЛДВОРУУД:")
        for finding in insights['key_findings']:
            lines.append(f"   {finding}")
        
        lines.append("")
        
        # Feature importance insight
        lines.append("[DATA] ШИНЖ ЧАНАРЫН АЧ ХОЛБОГДОЛ:")
        lines.append(f"   {insights['feature_importance']['insight']}")
        
        # Interactions
        if insights['interactions']['insight']:
            lines.append("")
            lines.append("[LINK] ХАРИЛЦАН ҮЙЛЧЛЭЛ:")
            lines.append(f"   {insights['interactions']['insight']}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def explain_single_prediction(
        self,
        shap_values: np.ndarray,
        X_single: np.ndarray,
        feature_names: List[str],
        base_value: float,
        prediction: float
    ) -> Dict[str, Any]:
        """Нэг таамаглалын тайлбар үүсгэх."""
        # Sorted by absolute SHAP value
        sorted_idx = np.argsort(np.abs(shap_values))[::-1]
        
        top_positive = []
        top_negative = []
        
        for idx in sorted_idx:
            item = {
                'feature': feature_names[idx],
                'value': float(X_single[idx]),
                'shap_value': float(shap_values[idx]),
                'contribution': 'increases' if shap_values[idx] > 0 else 'decreases'
            }
            
            if shap_values[idx] > 0:
                top_positive.append(item)
            else:
                top_negative.append(item)
        
        # Natural language explanation
        explanation_parts = []
        
        if top_positive:
            pos_features = [f"'{f['feature']}'={f['value']:.2f}" for f in top_positive[:3]]
            explanation_parts.append(f"Таамаглалыг нэмэгдүүлж буй: {', '.join(pos_features)}")
        
        if top_negative:
            neg_features = [f"'{f['feature']}'={f['value']:.2f}" for f in top_negative[:3]]
            explanation_parts.append(f"Таамаглалыг бууруулж буй: {', '.join(neg_features)}")
        
        return {
            'base_value': float(base_value),
            'prediction': float(prediction),
            'top_positive_factors': top_positive[:5],
            'top_negative_factors': top_negative[:5],
            'natural_language_explanation': " | ".join(explanation_parts),
            'total_positive_contribution': float(sum(f['shap_value'] for f in top_positive)),
            'total_negative_contribution': float(sum(f['shap_value'] for f in top_negative))
        }
