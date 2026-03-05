"""
Random Forest Model Wrapper - Random Forest загварын боодол
============================================================

Random Forest загварт зориулсан нэгдсэн интерфейстэй боодол.

Зохиогч: XAI-SHAP Framework
"""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.models.base_model import BaseModel


class RandomForestModel(BaseModel):
    """
    Random Forest загварын боодол - нэгдсэн интерфейстэй.
    
    Random Forest нь олон decision tree-г нэгтгэсэн
    найдвартай таамаглал хийдэг ensemble арга юм.
    
    Жишээ:
        >>> model = RandomForestModel({'n_estimators': 100})
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Random Forest загварыг эхлүүлэх.
        
        Параметрүүд:
            config: RF параметрүүдтэй загварын тохиргоо
        """
        super().__init__(config)
        self._task_type = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'RandomForestModel':
        """
        Random Forest загварыг сургалтын өгөгдөлд тохируулах.
        
        Параметрүүд:
            X: Сургалтын шинж чанарууд (features)
            y: Сургалтын шошго (labels)
            **kwargs: Нэмэлт параметрүүд
            
        Буцаах:
            self
        """
        # Даалгаврын төрлийг тодорхойлох
        n_classes = len(np.unique(y))
        self._task_type = 'classification' if n_classes <= 20 else 'regression'
        
        # Параметрүүдийг бүрдүүлэх
        params = {
            'n_estimators': self.config.get('n_estimators', 100),
            'max_depth': self.config.get('max_depth', 10),
            'min_samples_split': self.config.get('min_samples_split', 2),
            'min_samples_leaf': self.config.get('min_samples_leaf', 1),
            'random_state': self.config.get('random_state', 42),
            'n_jobs': -1,
            'verbose': 0
        }
        
        # Загварыг үүсгэж сургах
        if self._task_type == 'classification':
            self._model = RandomForestClassifier(**params)
        else:
            self._model = RandomForestRegressor(**params)
        
        self._model.fit(X, y)
        self._is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Таамаглал хийх."""
        if not self._is_fitted:
            raise ValueError("Загвар сургагдаагүй байна. Эхлээд fit() дуудна уу.")
        return self._model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Ангиллын магадлалыг таамаглах."""
        if not self._is_fitted:
            raise ValueError("Загвар сургагдаагүй байна. Эхлээд fit() дуудна уу.")
        if self._task_type != 'classification':
            return None
        return self._model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Random Forest-н feature importance авах."""
        if not self._is_fitted:
            return None
        return self._model.feature_importances_
    
    @property
    def model(self):
        """Дотоод sklearn загварт хандах."""
        return self._model
