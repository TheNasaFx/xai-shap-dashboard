"""
XGBoost Model Wrapper - XGBoost загварын боодол
================================================

XGBoost загварт зориулсан нэгдсэн интерфейстэй боодол.

Зохиогч: XAI-SHAP Framework
"""

from typing import Any, Dict, Optional
import numpy as np

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from src.models.base_model import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost загварын боодол - нэгдсэн интерфейстэй.
    
    XGBoost нь хүснэгтэн өгөгдөлд онцгой тохиромжтой
    хүчирхэг gradient boosting алгоритм юм.
    
    Жишээ:
        >>> model = XGBoostModel({'n_estimators': 100, 'max_depth': 6})
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        XGBoost загварыг эхлүүлэх.
        
        Параметрүүд:
            config: XGBoost параметрүүдтэй загварын тохиргоо
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost суулгаагүй байна. Дараах командыг ажиллуулна уу: pip install xgboost"
            )
        super().__init__(config)
        self._task_type = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
        **kwargs
    ) -> 'XGBoostModel':
        """
        XGBoost загварыг сургалтын өгөгдөлд тохируулах.
        
        Параметрүүд:
            X: Сургалтын шинж чанарууд (features)
            y: Сургалтын шошго (labels)
            eval_set: Эрт зогсоох үнэлгээний олонлог
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
            'max_depth': self.config.get('max_depth', 6),
            'learning_rate': self.config.get('learning_rate', 0.1),
            'random_state': self.config.get('random_state', 42),
            'n_jobs': -1,
            'verbosity': 0  # Анхааруулгыг хаах
        }
        
        # Даалгаврын төрлөөс хамаарч загварыг үүсгэх
        if self._task_type == 'classification':
            if n_classes == 2:
                params['objective'] = 'binary:logistic'
                params['eval_metric'] = 'logloss'
            else:
                params['objective'] = 'multi:softprob'
                params['eval_metric'] = 'mlogloss'
            self._model = xgb.XGBClassifier(**params)
        else:
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'rmse'
            self._model = xgb.XGBRegressor(**params)
        
        # Загварыг сургах
        fit_params = {}
        if eval_set:
            fit_params['eval_set'] = eval_set
            fit_params['verbose'] = False
        
        self._model.fit(X, y, **fit_params)
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
        """XGBoost-н feature importance авах."""
        if not self._is_fitted:
            return None
        return self._model.feature_importances_
    
    @property
    def model(self):
        """Дотоод XGBoost загварт хандах."""
        return self._model
