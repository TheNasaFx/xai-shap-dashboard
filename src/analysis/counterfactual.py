"""
Counterfactual Generator - "Юу өөрчлөгдвөл үр дүн өөрчлөгдөх вэ?"
==================================================================

Загварын таамаглалыг өөрчлөхөд ямар оролтыг хэрхэн
өөрчлөх шаардлагатайг олж, action-oriented тайлбар өгнө.

Жишээ: "Зээл батлуулахын тулд орлогоо $5,000 нэмэх хэрэгтэй"

Зохиогч: XAI-SHAP Framework
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Counterfactual:
    """Counterfactual тайлбарын класс."""
    original: np.ndarray
    counterfactual: np.ndarray
    original_prediction: float
    counterfactual_prediction: float
    changes: Dict[str, Dict[str, float]]
    distance: float
    validity: bool
    feature_names: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary руу хөрвүүлэх."""
        return {
            'original': self.original.tolist(),
            'counterfactual': self.counterfactual.tolist(),
            'original_prediction': self.original_prediction,
            'counterfactual_prediction': self.counterfactual_prediction,
            'changes': self.changes,
            'distance': self.distance,
            'validity': self.validity,
            'num_changes': len(self.changes)
        }
    
    def explain(self) -> str:
        """Хэрэглэгчид ойлгомжтой тайлбар."""
        if not self.validity:
            return "Counterfactual олдсонгүй - энэ таамаглалыг өөрчлөх боломжгүй байж магадгүй."
        
        lines = ["[COUNTERFACTUAL] ТАЙЛБАР:", ""]
        lines.append(f"Анхны таамаглал: {self.original_prediction:.3f}")
        lines.append(f"Шинэ таамаглал: {self.counterfactual_prediction:.3f}")
        lines.append("")
        lines.append("Хийх өөрчлөлтүүд:")
        
        for feature, change in self.changes.items():
            direction = "+" if change['change'] > 0 else "-"
            lines.append(
                f"  {direction} {feature}: {change['original']:.2f} -> {change['new']:.2f} "
                f"({'+' if change['change'] > 0 else ''}{change['change']:.2f})"
            )
        
        return "\n".join(lines)


class CounterfactualGenerator:
    """
    Counterfactual тайлбар үүсгэгч.
    
    Аргууд:
    - Gradient-based optimization
    - Genetic algorithm
    - Feature perturbation
    - DiCE-inspired diverse counterfactuals
    
    Жишээ:
        >>> generator = CounterfactualGenerator(model, feature_names)
        >>> cf = generator.generate(
        ...     instance=X_test[0],
        ...     target_class=1,
        ...     max_changes=3
        ... )
        >>> print(cf.explain())
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        categorical_features: Optional[List[str]] = None,
        immutable_features: Optional[List[str]] = None
    ):
        """
        CounterfactualGenerator эхлүүлэх.
        
        Параметрүүд:
            model: Сургасан загвар
            feature_names: Feature нэрсийн жагсаалт
            feature_ranges: Feature бүрийн утгын муж {feature: (min, max)}
            categorical_features: Категорийн features
            immutable_features: Өөрчлөх боломжгүй features (жнь: нас)
        """
        self.model = model
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.feature_ranges = feature_ranges or {}
        self.categorical_features = set(categorical_features or [])
        self.immutable_features = set(immutable_features or [])
        
    def generate(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        target_threshold: Optional[float] = None,
        max_changes: int = 5,
        method: str = 'optimization',
        n_counterfactuals: int = 1
    ) -> Union[Counterfactual, List[Counterfactual]]:
        """
        Counterfactual үүсгэх.
        
        Параметрүүд:
            instance: Тайлбарлах instance
            target_class: Зорилтот анги (classification)
            target_threshold: Зорилтот босго (regression)
            max_changes: Хамгийн их өөрчлөлтийн тоо
            method: 'optimization', 'genetic', 'perturbation'
            n_counterfactuals: Үүсгэх counterfactual тоо
            
        Буцаах:
            Counterfactual эсвэл Counterfactual-ийн жагсаалт
        """
        if method == 'optimization':
            cf = self._optimization_based(
                instance, target_class, target_threshold, max_changes
            )
        elif method == 'genetic':
            cf = self._genetic_algorithm(
                instance, target_class, target_threshold, max_changes
            )
        elif method == 'perturbation':
            cf = self._feature_perturbation(
                instance, target_class, target_threshold, max_changes
            )
        else:
            raise ValueError(f"Тодорхойгүй арга: {method}")
        
        if n_counterfactuals > 1:
            return self._generate_diverse(
                instance, target_class, target_threshold, 
                max_changes, n_counterfactuals
            )
        
        return cf
    
    def _optimization_based(
        self,
        instance: np.ndarray,
        target_class: Optional[int],
        target_threshold: Optional[float],
        max_changes: int
    ) -> Counterfactual:
        """Optimization-д суурилсан counterfactual."""
        original = instance.copy()
        original_pred = self._get_prediction(original)
        
        # Зорилтот утга тодорхойлох
        if target_class is not None:
            # Classification: зорилтот анги руу
            target_pred = 1.0 if target_class == 1 else 0.0
        elif target_threshold is not None:
            target_pred = target_threshold
        else:
            # Эсрэг таамаглал руу
            target_pred = 1.0 - original_pred
        
        # Loss function
        def loss(x):
            # Prediction loss
            pred = self._get_prediction(x)
            pred_loss = (pred - target_pred) ** 2
            
            # Distance loss (зөвхөн өөрчлөгдөх боломжтой features)
            diff = x - original
            for i, name in enumerate(self.feature_names):
                if name in self.immutable_features:
                    diff[i] = 0
            
            distance_loss = np.sum(diff ** 2)
            
            # Sparsity loss (цөөн өөрчлөлт)
            sparsity_loss = np.sum(np.abs(diff) > 0.01)
            
            return pred_loss * 100 + distance_loss * 0.1 + sparsity_loss * 0.5
        
        # Bounds тодорхойлох
        bounds = []
        for i, name in enumerate(self.feature_names):
            if name in self.immutable_features:
                bounds.append((original[i], original[i]))  # Өөрчлөхгүй
            elif name in self.feature_ranges:
                bounds.append(self.feature_ranges[name])
            else:
                # Default: ±50% утга, 0 бол ±1 хүрээ
                val = original[i]
                if abs(val) < 1e-10:
                    bounds.append((-1.0, 1.0))
                elif val > 0:
                    bounds.append((val * 0.5, val * 1.5))
                else:
                    bounds.append((val * 1.5, val * 0.5))
        
        # Optimization
        result = minimize(
            loss,
            original,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        counterfactual = result.x
        cf_pred = self._get_prediction(counterfactual)
        
        # Өөрчлөлтүүд
        changes = self._compute_changes(original, counterfactual, max_changes)
        
        # Хүчинтэй эсэх шалгах
        if target_class is not None:
            validity = (cf_pred > 0.5) == (target_class == 1)
        else:
            validity = abs(cf_pred - target_pred) < abs(original_pred - target_pred)
        
        return Counterfactual(
            original=original,
            counterfactual=counterfactual,
            original_prediction=float(original_pred),
            counterfactual_prediction=float(cf_pred),
            changes=changes,
            distance=float(np.sqrt(np.sum((counterfactual - original) ** 2))),
            validity=validity,
            feature_names=self.feature_names
        )
    
    def _genetic_algorithm(
        self,
        instance: np.ndarray,
        target_class: Optional[int],
        target_threshold: Optional[float],
        max_changes: int,
        population_size: int = 50,
        generations: int = 100
    ) -> Counterfactual:
        """Genetic algorithm-д суурилсан counterfactual."""
        original = instance.copy()
        original_pred = self._get_prediction(original)
        
        target_pred = 1.0 if target_class == 1 else 0.0 if target_class == 0 else (target_threshold or 1.0 - original_pred)
        
        np.random.seed(42)
        
        # Эхний population
        population = [original.copy() for _ in range(population_size)]
        for i in range(1, population_size):
            # Random perturbation
            mask = np.random.random(self.n_features) < 0.3
            for j, name in enumerate(self.feature_names):
                if mask[j] and name not in self.immutable_features:
                    population[i][j] *= np.random.uniform(0.8, 1.2)
        
        best_cf = None
        best_fitness = float('inf')
        
        for gen in range(generations):
            # Fitness тооцох
            fitness_scores = []
            for individual in population:
                pred = self._get_prediction(individual)
                pred_fitness = abs(pred - target_pred)
                distance_fitness = np.sqrt(np.sum((individual - original) ** 2))
                sparsity_fitness = np.sum(np.abs(individual - original) > 0.01)
                
                total_fitness = pred_fitness * 10 + distance_fitness * 0.1 + sparsity_fitness * 0.5
                fitness_scores.append(total_fitness)
                
                if total_fitness < best_fitness and pred_fitness < 0.3:
                    best_fitness = total_fitness
                    best_cf = individual.copy()
            
            # Selection
            sorted_idx = np.argsort(fitness_scores)
            survivors = [population[i] for i in sorted_idx[:population_size // 2]]
            
            # Crossover and mutation
            new_population = survivors.copy()
            while len(new_population) < population_size:
                parent1, parent2 = np.random.choice(len(survivors), 2, replace=False)
                child = survivors[parent1].copy()
                
                # Crossover
                crossover_mask = np.random.random(self.n_features) < 0.5
                child[crossover_mask] = survivors[parent2][crossover_mask]
                
                # Mutation
                mutation_mask = np.random.random(self.n_features) < 0.1
                for j in range(self.n_features):
                    if mutation_mask[j] and self.feature_names[j] not in self.immutable_features:
                        child[j] *= np.random.uniform(0.9, 1.1)
                
                new_population.append(child)
            
            population = new_population
        
        if best_cf is None:
            best_cf = population[np.argmin(fitness_scores)]
        
        cf_pred = self._get_prediction(best_cf)
        changes = self._compute_changes(original, best_cf, max_changes)
        
        if target_class is not None:
            validity = (cf_pred > 0.5) == (target_class == 1)
        else:
            validity = abs(cf_pred - target_pred) < abs(original_pred - target_pred)
        
        return Counterfactual(
            original=original,
            counterfactual=best_cf,
            original_prediction=float(original_pred),
            counterfactual_prediction=float(cf_pred),
            changes=changes,
            distance=float(np.sqrt(np.sum((best_cf - original) ** 2))),
            validity=validity,
            feature_names=self.feature_names
        )
    
    def _feature_perturbation(
        self,
        instance: np.ndarray,
        target_class: Optional[int],
        target_threshold: Optional[float],
        max_changes: int
    ) -> Counterfactual:
        """Feature perturbation-д суурилсан counterfactual."""
        original = instance.copy()
        original_pred = self._get_prediction(original)
        
        target_pred = 1.0 if target_class == 1 else 0.0 if target_class == 0 else (target_threshold or 1.0 - original_pred)
        
        # Feature бүрийн нөлөөг тооцох
        feature_effects = []
        for i, name in enumerate(self.feature_names):
            if name in self.immutable_features:
                feature_effects.append((i, 0))
                continue
            
            # +10% өөрчлөлтийн нөлөө
            perturbed = original.copy()
            perturbed[i] *= 1.1
            new_pred = self._get_prediction(perturbed)
            effect = new_pred - original_pred
            feature_effects.append((i, effect))
        
        # Нөлөөгөөр эрэмбэлэх
        if target_pred > original_pred:
            # Нэмэгдүүлэх чиглэлд
            feature_effects.sort(key=lambda x: -x[1])
        else:
            # Бууруулах чиглэлд
            feature_effects.sort(key=lambda x: x[1])
        
        # Top features өөрчлөх
        counterfactual = original.copy()
        changes_made = 0
        
        for idx, effect in feature_effects:
            if changes_made >= max_changes:
                break
            if abs(effect) < 0.001:
                continue
            
            # Өөрчлөлтийн хэмжээг тооцох
            current_pred = self._get_prediction(counterfactual)
            needed_change = target_pred - current_pred
            
            if abs(needed_change) < 0.05:
                break
            
            # Feature өөрчлөх
            if needed_change > 0 and effect > 0:
                multiplier = 1 + min(0.5, abs(needed_change) / abs(effect) * 0.1)
            elif needed_change < 0 and effect < 0:
                multiplier = 1 + min(0.5, abs(needed_change) / abs(effect) * 0.1)
            else:
                multiplier = 1 - min(0.5, abs(needed_change) / abs(effect) * 0.1)
            
            counterfactual[idx] *= multiplier
            changes_made += 1
        
        cf_pred = self._get_prediction(counterfactual)
        changes = self._compute_changes(original, counterfactual, max_changes)
        
        if target_class is not None:
            validity = (cf_pred > 0.5) == (target_class == 1)
        else:
            validity = abs(cf_pred - target_pred) < abs(original_pred - target_pred)
        
        return Counterfactual(
            original=original,
            counterfactual=counterfactual,
            original_prediction=float(original_pred),
            counterfactual_prediction=float(cf_pred),
            changes=changes,
            distance=float(np.sqrt(np.sum((counterfactual - original) ** 2))),
            validity=validity,
            feature_names=self.feature_names
        )
    
    def _generate_diverse(
        self,
        instance: np.ndarray,
        target_class: Optional[int],
        target_threshold: Optional[float],
        max_changes: int,
        n_counterfactuals: int
    ) -> List[Counterfactual]:
        """Олон янзын counterfactual үүсгэх."""
        counterfactuals = []
        methods = ['optimization', 'genetic', 'perturbation']
        
        # Өөр өөр аргаар үүсгэх
        for i in range(n_counterfactuals):
            method = methods[i % len(methods)]
            try:
                cf = self.generate(
                    instance, target_class, target_threshold,
                    max_changes, method, n_counterfactuals=1
                )
                counterfactuals.append(cf)
            except Exception as e:
                logger.warning(f"Counterfactual {i} үүсгэхэд алдаа: {e}")
        
        # Давхцсан цэвэрлэх
        unique_cfs = []
        for cf in counterfactuals:
            is_unique = True
            for existing in unique_cfs:
                if np.allclose(cf.counterfactual, existing.counterfactual, rtol=0.05):
                    is_unique = False
                    break
            if is_unique:
                unique_cfs.append(cf)
        
        return unique_cfs
    
    def _get_prediction(self, x: np.ndarray) -> float:
        """Загварын таамаглал авах."""
        x_2d = x.reshape(1, -1)
        
        if hasattr(self.model, 'predict_proba'):
            pred = self.model.predict_proba(x_2d)[0]
            return pred[1] if len(pred) > 1 else pred[0]
        else:
            return float(self.model.predict(x_2d)[0])
    
    def _compute_changes(
        self,
        original: np.ndarray,
        counterfactual: np.ndarray,
        max_changes: int
    ) -> Dict[str, Dict[str, float]]:
        """Өөрчлөлтүүдийг тооцох."""
        changes = {}
        diffs = []
        
        for i, name in enumerate(self.feature_names):
            diff = counterfactual[i] - original[i]
            if abs(diff) > 0.001:
                diffs.append((name, original[i], counterfactual[i], diff, abs(diff)))
        
        # Том өөрчлөлтүүдийг эхэнд
        diffs.sort(key=lambda x: -x[4])
        
        for name, orig, new, change, _ in diffs[:max_changes]:
            changes[name] = {
                'original': float(orig),
                'new': float(new),
                'change': float(change),
                'percent_change': float(change / orig * 100) if orig != 0 else 0
            }
        
        return changes
    
    def what_if_analysis(
        self,
        instance: np.ndarray,
        feature_name: str,
        values_range: Optional[Tuple[float, float]] = None,
        n_points: int = 20
    ) -> Dict[str, Any]:
        """
        "What-if" шинжилгээ - нэг feature өөрчлөхөд таамаглал хэрхэн өөрчлөгдөх.
        
        Параметрүүд:
            instance: Анхны instance
            feature_name: Өөрчлөх feature
            values_range: Утгын муж (min, max)
            n_points: Цэгүүдийн тоо
            
        Буцаах:
            What-if шинжилгээний үр дүн
        """
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature олдсонгүй: {feature_name}")
        
        feature_idx = self.feature_names.index(feature_name)
        original_value = instance[feature_idx]
        original_pred = self._get_prediction(instance)
        
        if values_range is None:
            values_range = (original_value * 0.5, original_value * 1.5)
        
        test_values = np.linspace(values_range[0], values_range[1], n_points)
        predictions = []
        
        for val in test_values:
            modified = instance.copy()
            modified[feature_idx] = val
            predictions.append(self._get_prediction(modified))
        
        # Threshold олох
        threshold_value = None
        for i in range(len(predictions) - 1):
            if (predictions[i] < 0.5 and predictions[i + 1] >= 0.5) or \
               (predictions[i] >= 0.5 and predictions[i + 1] < 0.5):
                # Linear interpolation
                t = (0.5 - predictions[i]) / (predictions[i + 1] - predictions[i])
                threshold_value = test_values[i] + t * (test_values[i + 1] - test_values[i])
                break
        
        return {
            'feature_name': feature_name,
            'original_value': float(original_value),
            'original_prediction': float(original_pred),
            'test_values': test_values.tolist(),
            'predictions': predictions,
            'threshold_value': float(threshold_value) if threshold_value else None,
            'sensitivity': float(np.std(predictions)),
            'min_prediction': float(min(predictions)),
            'max_prediction': float(max(predictions)),
            'prediction_range': float(max(predictions) - min(predictions))
        }
