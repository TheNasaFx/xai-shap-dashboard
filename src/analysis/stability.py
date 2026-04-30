"""
Stability Analyzer - SHAP утгын тогтвортой байдлын шинжилгээ
=============================================================

SHAP утгууд найдвартай, тогтвортой эсэхийг шинжлэх.
Bootstrap sampling болон Cross-validation аргуудыг ашиглана.

Зохиогч: XAI-SHAP Framework
"""

import logging
from typing import Any, Dict, List, Optional, Callable
import numpy as np
import pandas as pd
from scipy import stats
import warnings
from src.utils.helpers import create_shap_explainer, extract_shap_values

logger = logging.getLogger(__name__)


class StabilityAnalyzer:
    """
    SHAP утгын тогтвортой байдлын шинжилгээний класс.
    
    Шинжилгээний аргууд:
    - Bootstrap sampling: Өгөгдлийн дэд түүвэрлэлтээр SHAP-ийн итгэлцлийн интервал
    - Cross-validation: Fold-уудын хооронд SHAP-ийн тогтвортой байдал
    - Perturbation: Оролтын өөрчлөлтөөс SHAP-ийн мэдрэмж
    - Consistency: Өөр explainer-үүдийн хоорондын нийцтэй байдал
    
    Жишээ:
        >>> analyzer = StabilityAnalyzer()
        >>> stability = analyzer.analyze(
        ...     model=model,
        ...     X=X_test,
        ...     feature_names=feature_names,
        ...     method='bootstrap',
        ...     n_iterations=100
        ... )
        >>> print(stability['confidence_intervals'])
    """
    
    def __init__(self, config=None):
        """
        StabilityAnalyzer эхлүүлэх.
        
        Параметрүүд:
            config: Тохиргооны менежер instance
        """
        self.config = config
        self.default_n_iterations = 100
        self.confidence_level = 0.95
        self.random_state = 42
        
    def analyze(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        shap_values: Optional[np.ndarray] = None,
        method: str = 'bootstrap',
        n_iterations: int = 100,
        sample_fraction: float = 0.8
    ) -> Dict[str, Any]:
        """
        SHAP утгын тогтвортой байдлыг шинжлэх.
        
        Параметрүүд:
            model: Сургасан загвар
            X: Feature массив
            feature_names: Feature нэрсийн жагсаалт
            shap_values: Өмнө тооцсон SHAP утгууд (заавал биш)
            method: 'bootstrap', 'cv', 'perturbation'
            n_iterations: Давталтын тоо
            sample_fraction: Түүвэрлэлтийн хувь
            
        Буцаах:
            Тогтвортой байдлын шинжилгээний үр дүн
        """
        logger.info(f"SHAP тогтвортой байдлын шинжилгээ: {method}")
        
        if method == 'bootstrap':
            return self._bootstrap_analysis(
                model, X, feature_names, n_iterations, sample_fraction,
                shap_values=shap_values
            )
        elif method == 'cv':
            return self._cross_validation_analysis(
                model, X, feature_names, n_folds=5
            )
        elif method == 'perturbation':
            return self._perturbation_analysis(
                model, X, feature_names, shap_values
            )
        else:
            raise ValueError(f"Тодорхойгүй арга: {method}")
    
    def _bootstrap_analysis(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        n_iterations: int,
        sample_fraction: float,
        shap_values: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Bootstrap sampling ашиглан тогтвортой байдал шинжлэх."""
        n_samples = len(X)
        sample_size = int(n_samples * sample_fraction)
        n_features = len(feature_names)
        
        # Bootstrap SHAP утгуудыг хадгалах
        bootstrap_shap_means = np.zeros((n_iterations, n_features))
        bootstrap_shap_stds = np.zeros((n_iterations, n_features))
        
        np.random.seed(self.random_state)
        
        # Өмнө тооцсон SHAP утгууд байвал тэдгээрийг resample хийнэ
        use_precomputed = shap_values is not None
        if use_precomputed:
            shap_values = np.asarray(shap_values, dtype=np.float64)
            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1] if shap_values.shape[2] == 2 else shap_values.mean(axis=2)
            logger.info(f"Өмнө тооцсон SHAP утгууд дээр bootstrap хийж байна ({n_iterations} давталт)...")
        else:
            logger.info(f"Bootstrap эхэлж байна ({n_iterations} давталт)...")
        
        for i in range(n_iterations):
            # Random sample
            indices = np.random.choice(n_samples, size=sample_size, replace=True)
            
            try:
                if use_precomputed:
                    # Өмнө тооцсон SHAP-аас resample
                    shap_sample = shap_values[indices]
                    bootstrap_shap_means[i] = np.abs(shap_sample).mean(axis=0)
                    bootstrap_shap_stds[i] = shap_sample.std(axis=0)
                else:
                    # SHAP дахин тооцох
                    X_sample = X[indices]
                    background = X_sample[:min(100, len(X_sample))]
                    explainer = create_shap_explainer(model, background)
                    shap_vals = extract_shap_values(explainer, X_sample)
                    
                    if len(shap_vals.shape) == 3:
                        shap_vals = shap_vals[:, :, 1] if shap_vals.shape[2] == 2 else shap_vals.mean(axis=2)
                    
                    bootstrap_shap_means[i] = np.abs(shap_vals).mean(axis=0)
                    bootstrap_shap_stds[i] = shap_vals.std(axis=0)
                
            except Exception as e:
                logger.warning(f"Bootstrap {i} алдаа: {e}")
                bootstrap_shap_means[i] = np.nan
                bootstrap_shap_stds[i] = np.nan
            
            if (i + 1) % 20 == 0:
                logger.info(f"  Давталт {i + 1}/{n_iterations} дууссан")
        
        # NaN-гүй үр дүнгүүд
        valid_mask = ~np.isnan(bootstrap_shap_means).any(axis=1)
        valid_means = bootstrap_shap_means[valid_mask]
        
        if len(valid_means) < 10:
            return {
                'success': False,
                'error': 'Хангалттай bootstrap түүвэрлэлт амжилтгүй болсон'
            }
        
        # Статистик тооцох
        results = {
            'method': 'bootstrap',
            'n_iterations': n_iterations,
            'valid_iterations': int(valid_mask.sum()),
            'confidence_level': self.confidence_level,
            'feature_stability': {},
            'overall_stability_score': 0,
            'confidence_intervals': {},
            'rankings_stability': None
        }
        
        alpha = 1 - self.confidence_level
        
        for j, feature in enumerate(feature_names):
            feature_means = valid_means[:, j]
            
            # Итгэлцлийн интервал
            ci_low = np.percentile(feature_means, alpha/2 * 100)
            ci_high = np.percentile(feature_means, (1 - alpha/2) * 100)
            
            # Тогтвортой байдлын оноо (CV коэффициент)
            cv = feature_means.std() / (feature_means.mean() + 1e-10)
            stability_score = max(0, 1 - cv)
            
            results['feature_stability'][feature] = {
                'mean_importance': float(feature_means.mean()),
                'std': float(feature_means.std()),
                'cv': float(cv),
                'ci_low': float(ci_low),
                'ci_high': float(ci_high),
                'stability_score': float(stability_score),
                'is_stable': stability_score > 0.7
            }
            
            results['confidence_intervals'][feature] = {
                'mean': float(feature_means.mean()),
                'lower': float(ci_low),
                'upper': float(ci_high),
                'ci_width': float(ci_high - ci_low)
            }
        
        # Ranking тогтвортой байдал
        results['rankings_stability'] = self._analyze_ranking_stability(
            valid_means, feature_names
        )
        
        # Нийт оноо
        stability_scores = [v['stability_score'] for v in results['feature_stability'].values()]
        results['overall_stability_score'] = float(np.mean(stability_scores))
        
        # Дүгнэлт
        results['summary'] = self._generate_stability_summary(results)
        
        return results
    
    def _cross_validation_analysis(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        n_folds: int = 5
    ) -> Dict[str, Any]:
        """Cross-validation ашиглан тогтвортой байдал шинжлэх."""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        n_features = len(feature_names)
        
        fold_shap_means = np.zeros((n_folds, n_features))
        
        logger.info(f"{n_folds}-fold CV эхэлж байна...")
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_fold = X[test_idx]
            
            try:
                background = X[train_idx][:min(100, len(train_idx))]
                explainer = create_shap_explainer(model, background)
                shap_vals = extract_shap_values(explainer, X_fold)
                
                if len(shap_vals.shape) == 3:
                    shap_vals = shap_vals[:, :, 1] if shap_vals.shape[2] == 2 else shap_vals.mean(axis=2)
                
                fold_shap_means[fold_idx] = np.abs(shap_vals).mean(axis=0)
                
            except Exception as e:
                logger.warning(f"Fold {fold_idx} алдаа: {e}")
                fold_shap_means[fold_idx] = np.nan
        
        # Үр дүн
        results = {
            'method': 'cross_validation',
            'n_folds': n_folds,
            'feature_stability': {},
            'fold_agreement': {}
        }
        
        for j, feature in enumerate(feature_names):
            fold_means = fold_shap_means[:, j]
            valid_means = fold_means[~np.isnan(fold_means)]
            
            if len(valid_means) < 2:
                continue
            
            cv = valid_means.std() / (valid_means.mean() + 1e-10)
            
            results['feature_stability'][feature] = {
                'mean_importance': float(valid_means.mean()),
                'std_across_folds': float(valid_means.std()),
                'cv': float(cv),
                'stability_score': float(max(0, 1 - cv)),
                'fold_values': valid_means.tolist()
            }
        
        # Fold хоорондын эрэмбэлэлтийн нийцтэй байдал
        results['rankings_stability'] = self._analyze_ranking_stability(
            fold_shap_means, feature_names
        )
        
        # Нийт оноо
        stability_scores = [v['stability_score'] for v in results['feature_stability'].values()]
        results['overall_stability_score'] = float(np.mean(stability_scores)) if stability_scores else 0
        
        results['summary'] = self._generate_cv_summary(results)
        
        return results
    
    def _perturbation_analysis(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        shap_values: Optional[np.ndarray],
        noise_level: float = 0.01
    ) -> Dict[str, Any]:
        """Оролтын өөрчлөлтөөс SHAP-ийн мэдрэмжийг шинжлэх."""
        if shap_values is None:
            background = X[:min(100, len(X))]
            explainer = create_shap_explainer(model, background)
            shap_values = extract_shap_values(explainer, X)
            
            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1] if shap_values.shape[2] == 2 else shap_values.mean(axis=2)
        
        # Оролтод жижиг noise нэмэх
        np.random.seed(self.random_state)
        X_perturbed = X + np.random.normal(0, noise_level * X.std(axis=0), X.shape)
        
        # Perturbed SHAP тооцох
        try:
            background = X_perturbed[:min(100, len(X_perturbed))]
            explainer = create_shap_explainer(model, background)
            perturbed_shap = extract_shap_values(explainer, X_perturbed)
            
            if len(perturbed_shap.shape) == 3:
                perturbed_shap = perturbed_shap[:, :, 1] if perturbed_shap.shape[2] == 2 else perturbed_shap.mean(axis=2)
        except Exception as e:
            return {'success': False, 'error': str(e)}
        
        # Өөрчлөлтийн шинжилгээ
        results = {
            'method': 'perturbation',
            'noise_level': noise_level,
            'feature_sensitivity': {},
            'overall_robustness': 0
        }
        
        for j, feature in enumerate(feature_names):
            original = shap_values[:, j]
            perturbed = perturbed_shap[:, j]
            
            # Корреляци (их бол тогтвортой)
            corr, _ = stats.spearmanr(original, perturbed)
            
            # MAE (бага бол тогтвортой)
            mae = np.abs(original - perturbed).mean()
            relative_change = mae / (np.abs(original).mean() + 1e-10)
            
            robustness_score = corr * (1 - min(1, relative_change))
            
            results['feature_sensitivity'][feature] = {
                'correlation': float(corr),
                'mae': float(mae),
                'relative_change': float(relative_change),
                'robustness_score': float(max(0, robustness_score)),
                'is_robust': robustness_score > 0.8
            }
        
        # Нийт robustness
        robustness_scores = [v['robustness_score'] for v in results['feature_sensitivity'].values()]
        results['overall_robustness'] = float(np.mean(robustness_scores))
        
        results['summary'] = self._generate_perturbation_summary(results)
        
        return results
    
    def _analyze_ranking_stability(
        self,
        shap_matrices: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Feature эрэмбэлэлтийн тогтвортой байдал."""
        valid_matrices = shap_matrices[~np.isnan(shap_matrices).any(axis=1)]
        
        if len(valid_matrices) < 2:
            return {'stable': False, 'reason': 'Хангалттай өгөгдөл байхгүй'}
        
        # Давталт бүрт эрэмбэлэлт
        rankings = np.argsort(-valid_matrices, axis=1)  # Буурах дарааллаар
        
        # Kendall's W (concordance coefficient)
        n_iterations = len(rankings)
        n_features = len(feature_names)
        
        # Эрэмбийн нийлбэр feature бүрт
        rank_sums = rankings.sum(axis=0)
        mean_rank_sum = rank_sums.mean()
        
        # S статистик
        S = ((rank_sums - mean_rank_sum) ** 2).sum()
        
        # Kendall's W
        W = 12 * S / (n_iterations ** 2 * (n_features ** 3 - n_features))
        
        # Top-k consistency
        top_k = min(5, n_features)
        top_k_consistency = 0
        
        for i in range(len(rankings) - 1):
            top_k_set_1 = set(rankings[i][:top_k])
            top_k_set_2 = set(rankings[i + 1][:top_k])
            top_k_consistency += len(top_k_set_1 & top_k_set_2) / top_k
        
        top_k_consistency /= (len(rankings) - 1)
        
        # Feature бүрийн эрэмбийн хэлбэлзэл
        rank_volatility = {}
        for j, feature in enumerate(feature_names):
            feature_ranks = rankings[:, j] if rankings.ndim > 1 else [j]
            rank_volatility[feature] = {
                'mean_rank': float(np.mean([np.where(r == j)[0][0] for r in rankings]) + 1),
                'rank_std': float(np.std([np.where(r == j)[0][0] for r in rankings])),
                'most_common_rank': int(stats.mode([np.where(r == j)[0][0] for r in rankings], keepdims=False)[0]) + 1
            }
        
        return {
            'kendalls_w': float(W),
            'top_k_consistency': float(top_k_consistency),
            'rank_volatility': rank_volatility,
            'is_ranking_stable': W > 0.7 and top_k_consistency > 0.8,
            'interpretation': self._interpret_ranking_stability(W, top_k_consistency)
        }
    
    def _interpret_ranking_stability(self, W: float, top_k: float) -> str:
        """Эрэмбийн тогтвортой байдлын тайлбар."""
        if W > 0.9 and top_k > 0.9:
            return "Маш тогтвортой: Feature эрэмбэлэлт найдвартай"
        elif W > 0.7 and top_k > 0.8:
            return "Тогтвортой: Дээд features тогтвортой, бага ач холбогдолтой features хэлбэлздэг"
        elif W > 0.5:
            return "Дунд зэрэг: Зарим тогтворгүй байдал ажиглагдаж байна"
        else:
            return "Тогтворгүй: Feature эрэмбэлэлт найдваргүй, загвар эсвэл өгөгдлийг шалгах"
    
    def _generate_stability_summary(self, results: Dict[str, Any]) -> str:
        """Bootstrap шинжилгээний хураангуй."""
        lines = [
            "=" * 60,
            "         SHAP ТОГТВОРТОЙ БАЙДЛЫН ШИНЖИЛГЭЭ",
            "           (Bootstrap арга)",
            "=" * 60,
            "",
            f"[DATA] Давталтын тоо: {results['n_iterations']}",
            f"[DATA] Амжилттай давталт: {results['valid_iterations']}",
            f"[DATA] Итгэлцлийн түвшин: {results['confidence_level'] * 100:.0f}%",
            "",
            f"[SCORE] Нийт тогтвортой байдлын оноо: {results['overall_stability_score']:.2f}",
            ""
        ]
        
        # Feature тус бүрийн тогтвортой байдал
        stable_features = [f for f, v in results['feature_stability'].items() if v['is_stable']]
        unstable_features = [f for f, v in results['feature_stability'].items() if not v['is_stable']]
        
        lines.append(f"[OK] Тогтвортой features: {len(stable_features)}")
        lines.append(f"[!] Тогтворгүй features: {len(unstable_features)}")
        
        if unstable_features:
            lines.append(f"   └─ {', '.join(unstable_features[:5])}")
        
        # Ranking stability
        if results['rankings_stability']:
            rs = results['rankings_stability']
            lines.extend([
                "",
                f"[RANK] Эрэмбийн нийцтэй байдал (Kendall's W): {rs['kendalls_w']:.2f}",
                f"[RANK] Top-5 тогтвортой байдал: {rs['top_k_consistency']:.2f}",
                f"   {rs['interpretation']}"
            ])
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _generate_cv_summary(self, results: Dict[str, Any]) -> str:
        """CV шинжилгээний хураангуй."""
        return f"""
===============================================================
         SHAP ТОГТВОРТОЙ БАЙДЛЫН ШИНЖИЛГЭЭ
              (Cross-Validation арга)
===============================================================

[DATA] Fold тоо: {results['n_folds']}
[SCORE] Нийт тогтвортой байдлын оноо: {results['overall_stability_score']:.2f}

===============================================================
"""
    
    def _generate_perturbation_summary(self, results: Dict[str, Any]) -> str:
        """Perturbation шинжилгээний хураангуй."""
        robust = [f for f, v in results['feature_sensitivity'].items() if v['is_robust']]
        sensitive = [f for f, v in results['feature_sensitivity'].items() if not v['is_robust']]
        
        return f"""
===============================================================
         SHAP ROBUSTNESS ШИНЖИЛГЭЭ
            (Perturbation арга)
===============================================================

[DATA] Noise түвшин: {results['noise_level'] * 100:.1f}%
[SCORE] Нийт robustness оноо: {results['overall_robustness']:.2f}

[OK] Robust features: {len(robust)}
[!] Мэдрэг features: {len(sensitive)}

===============================================================
"""
