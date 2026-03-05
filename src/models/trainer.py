"""
Model Trainer - Загвар сургагч
===============================

Янз бүрийн black-box машин сургалтын загваруудыг
нэгдсэн тохиргоогоор сургах интерфейс.

Зохиогч: XAI-SHAP Framework
"""

import logging
from typing import Any, Dict, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Машин сургалтын загваруудыг сургах нэгдсэн интерфейс.
    
    Энэ класс дараах боломжуудыг олгоно:
    - Өөр өөр загварын төрлүүдэд нэгдсэн API
    - Тохиргоонд суурилсан hyperparameter удирдлага
    - Автомат даалгаврын төрөл тодорхойлолт (classification/regression)
    - Сургалтын metrics цуглуулалт
    
    Дэмжигдсэн загварууд:
    - XGBoost (xgboost)
    - Random Forest (random_forest)
    - Neural Network (neural_network)
    - LightGBM (lightgbm)
    - CatBoost (catboost)
    - Logistic Regression (logistic_regression)
    - Gradient Boosting (gradient_boosting)
    - Support Vector Machine (svm)
    - AdaBoost (adaboost)
    - Extra Trees (extra_trees)
    
    Жишээ:
        >>> trainer = ModelTrainer(config)
        >>> model = trainer.train(
        ...     model_type='xgboost',
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     n_estimators=100
        ... )
    """
    
    def __init__(self, config=None):
        """
        ModelTrainer эхлүүлэх.
        
        Параметрүүд:
            config: Тохиргооны менежер instance
        """
        self.config = config
        self._model = None
        self._training_history = {}
    
    def train(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        task_type: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Машин сургалтын загварыг сургах.
        
        Параметрүүд:
            model_type: Загварын төрөл ('xgboost', 'random_forest', гэх мэт)
            X_train: Сургалтын шинж чанарууд
            y_train: Сургалтын шошго
            X_test: Тест шинж чанарууд (эрт зогсооход, заавал биш)
            y_test: Тест шошго (заавал биш)
            task_type: 'classification' эсвэл 'regression' (автомат илрүүлэх)
            **kwargs: Загвар-тодорхой hyperparameter-ууд
            
        Буцаах:
            Сургасан загварын обьект
        """
        # Даалгаврын төрлийг автомат илрүүлэх
        if task_type is None:
            task_type = self._detect_task_type(y_train)
        
        logger.info(f"{task_type} даалгаварт {model_type} загварыг сургаж байна")
        
        # Config файлаас загварын тохиргоог авах
        model_config = {}
        if self.config:
            model_config = self.config.get(f"models.{model_type}", {})
        
        # Дамжуулсан kwargs-ээр дарах
        model_config.update(kwargs)
        
        # Загварыг үүсгэж сургах
        if model_type == 'xgboost':
            model = self._train_xgboost(
                X_train, y_train, X_test, y_test, task_type, model_config
            )
        elif model_type == 'random_forest':
            model = self._train_random_forest(
                X_train, y_train, task_type, model_config
            )
        elif model_type == 'neural_network':
            model = self._train_neural_network(
                X_train, y_train, X_test, y_test, task_type, model_config
            )
        elif model_type == 'lightgbm':
            model = self._train_lightgbm(
                X_train, y_train, X_test, y_test, task_type, model_config
            )
        elif model_type == 'catboost':
            model = self._train_catboost(
                X_train, y_train, X_test, y_test, task_type, model_config
            )
        elif model_type == 'logistic_regression':
            model = self._train_logistic_regression(
                X_train, y_train, task_type, model_config
            )
        elif model_type == 'gradient_boosting':
            model = self._train_gradient_boosting(
                X_train, y_train, task_type, model_config
            )
        elif model_type == 'svm':
            model = self._train_svm(
                X_train, y_train, task_type, model_config
            )
        elif model_type == 'adaboost':
            model = self._train_adaboost(
                X_train, y_train, task_type, model_config
            )
        elif model_type == 'extra_trees':
            model = self._train_extra_trees(
                X_train, y_train, task_type, model_config
            )
        else:
            raise ValueError(f"Тодорхойгүй загварын төрөл: {model_type}")
        
        self._model = model
        self._training_history['model_type'] = model_type
        self._training_history['task_type'] = task_type
        
        return model
    
    def _detect_task_type(self, y: np.ndarray) -> str:
        """Classification эсвэл regression эсэхийг тодорхойлох."""
        unique_values = np.unique(y)
        
        # Эвристик: хэрэв цөөн давтагдашгүй утга эсвэл бүхэл тоо бол classification
        if len(unique_values) <= 20:
            return 'classification'
        if y.dtype in [np.int32, np.int64]:
            return 'classification'
        
        return 'regression'
    
    def _train_xgboost(
        self,
        X_train, y_train, X_test, y_test, task_type, config
    ):
        """XGBoost загварыг сургах."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost суулгаагүй байна. Дараах командыг ажиллуулна уу: pip install xgboost")
        
        # Үндсэн параметрүүд
        params = {
            'n_estimators': config.get('n_estimators', 100),
            'max_depth': config.get('max_depth', 6),
            'learning_rate': config.get('learning_rate', 0.1),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        if task_type == 'classification':
            # Label-уудыг 0-ээс эхэлсэн дараалалтай болгох (XGBoost шаардлага)
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test) if y_test is not None else None
            
            n_classes = len(le.classes_)
            
            if n_classes < 2:
                raise ValueError(f"Classification-д хамгийн багадаа 2 класс шаардлагатай. Олдсон: {n_classes}")
            
            if n_classes == 2:
                params['objective'] = 'binary:logistic'
                params['eval_metric'] = 'logloss'
            else:
                params['objective'] = 'multi:softprob'
                params['eval_metric'] = 'mlogloss'
                params['num_class'] = n_classes
            
            model = xgb.XGBClassifier(**params)
            
            # Тест өгөгдөл байвал early stopping-тэй сургах
            if X_test is not None and y_test_encoded is not None:
                model.fit(
                    X_train, y_train_encoded,
                    eval_set=[(X_test, y_test_encoded)],
                    verbose=False
                )
            else:
                model.fit(X_train, y_train_encoded)
            
            # LabelEncoder-ийг хадгалах (predict үед хэрэгтэй)
            model._label_encoder = le
        else:
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'rmse'
            model = xgb.XGBRegressor(**params)
            
            # Тест өгөгдөл байвал early stopping-тэй сургах
            if X_test is not None and y_test is not None:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)
        
        logger.info(f"XGBoost загвар {params['n_estimators']} estimator-аар сургагдлаа")
        
        return model
    
    def _train_random_forest(
        self,
        X_train, y_train, task_type, config
    ):
        """Random Forest загварыг сургах."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        params = {
            'n_estimators': config.get('n_estimators', 100),
            'max_depth': config.get('max_depth', 10),
            'min_samples_split': config.get('min_samples_split', 2),
            'min_samples_leaf': config.get('min_samples_leaf', 1),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 0
        }
        
        if task_type == 'classification':
            model = RandomForestClassifier(**params)
        else:
            model = RandomForestRegressor(**params)
        
        model.fit(X_train, y_train)
        
        logger.info(f"Random Forest {params['n_estimators']} модоор сургагдлаа")
        
        return model
    
    def _train_neural_network(
        self,
        X_train, y_train, X_test, y_test, task_type, config
    ):
        """Neural Network загварыг sklearn MLPClassifier/Regressor ашиглан сургах."""
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        
        # Config-оос hidden layers задлах
        hidden_layers = config.get('hidden_layers', [128, 64, 32])
        if isinstance(hidden_layers, list):
            hidden_layers = tuple(hidden_layers)
        
        params = {
            'hidden_layer_sizes': hidden_layers,
            'activation': config.get('activation', 'relu'),
            'learning_rate_init': config.get('learning_rate', 0.001),
            'max_iter': config.get('epochs', 200),
            'random_state': 42,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10,
            'verbose': False
        }
        
        if task_type == 'classification':
            model = MLPClassifier(**params)
        else:
            model = MLPRegressor(**params)
        
        model.fit(X_train, y_train)
        
        logger.info(f"Neural Network {hidden_layers} давхаргуудаар сургагдлаа")
        
        return model
    
    def _train_lightgbm(
        self,
        X_train, y_train, X_test, y_test, task_type, config
    ):
        """LightGBM загварыг сургах."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM суулгаагүй байна. Дараах командыг ажиллуулна уу: pip install lightgbm")
        
        params = {
            'n_estimators': config.get('n_estimators', 100),
            'max_depth': config.get('max_depth', 6),
            'learning_rate': config.get('learning_rate', 0.1),
            'num_leaves': config.get('num_leaves', 31),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'force_col_wise': True  # Анхааруулгыг хаах
        }
        
        if task_type == 'classification':
            model = lgb.LGBMClassifier(**params)
        else:
            model = lgb.LGBMRegressor(**params)
        
        # Тест өгөгдөл байвал early stopping-тэй сургах
        if X_test is not None and y_test is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)]
            )
        else:
            model.fit(X_train, y_train)
        
        logger.info(f"LightGBM загвар {params['n_estimators']} estimator-аар сургагдлаа")
        
        return model
    
    def _train_catboost(
        self,
        X_train, y_train, X_test, y_test, task_type, config
    ):
        """CatBoost загварыг сургах."""
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError:
            raise ImportError("CatBoost суулгаагүй байна. Дараах командыг ажиллуулна уу: pip install catboost")
        
        params = {
            'iterations': config.get('n_estimators', 100),
            'depth': config.get('max_depth', 6),
            'learning_rate': config.get('learning_rate', 0.1),
            'random_seed': 42,
            'verbose': False,
            'allow_writing_files': False
        }
        
        if task_type == 'classification':
            model = CatBoostClassifier(**params)
        else:
            model = CatBoostRegressor(**params)
        
        # Тест өгөгдөл байвал early stopping-тэй сургах
        if X_test is not None and y_test is not None:
            model.fit(
                X_train, y_train,
                eval_set=(X_test, y_test),
                early_stopping_rounds=20
            )
        else:
            model.fit(X_train, y_train)
        
        logger.info(f"CatBoost загвар {params['iterations']} iteration-аар сургагдлаа")
        
        return model
    
    def _train_logistic_regression(
        self,
        X_train, y_train, task_type, config
    ):
        """Logistic Regression загварыг сургах."""
        from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        # Regularization төрөл
        penalty = config.get('penalty', 'l2')
        C = config.get('C', 1.0)
        
        if task_type == 'classification':
            # Шинэ sklearn version-д penalty параметр өөрчлөгдсөн
            params = {
                'C': C,
                'solver': 'lbfgs' if penalty == 'l2' else 'saga',
                'max_iter': config.get('max_iter', 1000),
                'random_state': 42
            }
            
            # l1_ratio ашиглах (penalty-ийн оронд)
            if penalty == 'l1':
                params['l1_ratio'] = 1.0
                params['penalty'] = 'elasticnet'
            elif penalty == 'elasticnet':
                params['l1_ratio'] = 0.5
                params['penalty'] = 'elasticnet'
            # l2 бол default тул тохируулах шаардлагагүй
            
            # Pipeline with scaling
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(**params))
            ])
        else:
            # Regression-д Ridge эсвэл ElasticNet ашиглах
            alpha = 1.0 / C if C > 0 else 1.0
            
            if penalty == 'l2':
                reg_model = Ridge(alpha=alpha, random_state=42)
            else:
                reg_model = ElasticNet(alpha=alpha, random_state=42, max_iter=config.get('max_iter', 1000))
            
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', reg_model)
            ])
        
        model.fit(X_train, y_train)
        
        logger.info(f"Logistic Regression загвар {penalty} penalty-тэй сургагдлаа")
        
        return model
    
    def _train_gradient_boosting(
        self,
        X_train, y_train, task_type, config
    ):
        """sklearn Gradient Boosting загварыг сургах."""
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        
        params = {
            'n_estimators': config.get('n_estimators', 100),
            'max_depth': config.get('max_depth', 3),
            'learning_rate': config.get('learning_rate', 0.1),
            'min_samples_split': config.get('min_samples_split', 2),
            'min_samples_leaf': config.get('min_samples_leaf', 1),
            'random_state': 42,
            'verbose': 0
        }
        
        if task_type == 'classification':
            model = GradientBoostingClassifier(**params)
        else:
            model = GradientBoostingRegressor(**params)
        
        model.fit(X_train, y_train)
        
        logger.info(f"Gradient Boosting загвар {params['n_estimators']} estimator-аар сургагдлаа")
        
        return model
    
    def _train_svm(
        self,
        X_train, y_train, task_type, config
    ):
        """Support Vector Machine загварыг сургах."""
        from sklearn.svm import SVC, SVR
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        params = {
            'kernel': config.get('kernel', 'rbf'),
            'C': config.get('C', 1.0),
            'gamma': config.get('gamma', 'scale'),
            'random_state': 42
        }
        
        if task_type == 'classification':
            params['probability'] = True  # SHAP-д шаардлагатай
            svm_model = SVC(**params)
        else:
            svm_model = SVR(**params)
        
        # Pipeline with scaling (SVM scaling-д мэдрэмтгий)
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', svm_model)
        ])
        
        model.fit(X_train, y_train)
        
        logger.info(f"SVM загвар {params['kernel']} kernel-тэй сургагдлаа")
        
        return model
    
    def _train_adaboost(
        self,
        X_train, y_train, task_type, config
    ):
        """AdaBoost загварыг сургах."""
        from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        
        n_estimators = config.get('n_estimators', 50)
        learning_rate = config.get('learning_rate', 1.0)
        max_depth = config.get('max_depth', 3)
        
        if task_type == 'classification':
            base_estimator = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            model = AdaBoostClassifier(
                estimator=base_estimator,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42
            )
        else:
            base_estimator = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
            model = AdaBoostRegressor(
                estimator=base_estimator,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42
            )
        
        model.fit(X_train, y_train)
        
        logger.info(f"AdaBoost загвар {n_estimators} estimator-аар сургагдлаа")
        
        return model
    
    def _train_extra_trees(
        self,
        X_train, y_train, task_type, config
    ):
        """Extra Trees загварыг сургах."""
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        
        params = {
            'n_estimators': config.get('n_estimators', 100),
            'max_depth': config.get('max_depth', None),
            'min_samples_split': config.get('min_samples_split', 2),
            'min_samples_leaf': config.get('min_samples_leaf', 1),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 0
        }
        
        if task_type == 'classification':
            model = ExtraTreesClassifier(**params)
        else:
            model = ExtraTreesRegressor(**params)
        
        model.fit(X_train, y_train)
        
        logger.info(f"Extra Trees загвар {params['n_estimators']} модоор сургагдлаа")
        
        return model
    
    @property
    def model(self):
        """Сургасан загварыг авах."""
        return self._model
    
    @property
    def training_history(self) -> Dict[str, Any]:
        """Сургалтын түүх болон metadata авах."""
        return self._training_history
