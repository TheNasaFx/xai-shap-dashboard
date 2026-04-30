"""
XAI Framework - Үндсэн Эхлэх Цэг
=================================

Энэ нь хар хайрцагт загваруудын тайлбарлах боломжтой AI шинжилгээний
бүх бүрэлдэхүүн хэсгүүдийг зохицуулдаг үндсэн framework класс юм.

Framework нь дараахыг хангана:
- Өгөгдөл боловсруулах, загвар сургах, тайлбарлахад зориулсан нэгдсэн интерфейс
- SHAP-д суурилсан шинж чанарын ач холбогдлын шинжилгээ
- Визуал аналитик dashboard үүсгэх
- Хариуцлагатай AI үнэлгээ

Зохиогч: XAI-SHAP Framework
"""

import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd

from src.core.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class XAIFramework:
    """
    XAI-SHAP Визуал Аналитикийн үндсэн framework класс.
    
    Энэ класс нь дараах зориулалтаар үндсэн эхлэх цэг болж үйлчилнэ:
    1. Хазайлт илрүүлэлттэй өгөгдөл ачаалах, урьдчилан боловсруулах
    2. Хар хайрцагт загварууд сургах (XGBoost, RF, Neural Networks)
    3. SHAP тайлбарууд үүсгэх (локал ба глобал)
    4. Интерактив визуал аналитик үүсгэх
    5. Загварын шударга байдал ба ил тод байдлыг үнэлэх
    
    Архитектур:
    ```
    ┌─────────────────────────────────────────────────────────┐
    │                     XAIFramework                        │
    ├─────────────────────────────────────────────────────────┤
    │  DataProcessor → ModelTrainer → SHAPExplainer → Viz    │
    │                        ↓                                │
    │              ResponsibleAI Evaluator                    │
    └─────────────────────────────────────────────────────────┘
    ```
    
    Жишээ:
        >>> from src.core.framework import XAIFramework
        >>> 
        >>> # Framework эхлүүлэх
        >>> framework = XAIFramework()
        >>> 
        >>> # Өгөгдөл ачаалж боловсруулах
        >>> framework.load_data("data/sample/dataset.csv", target="target")
        >>> 
        >>> # Загвар сургах
        >>> framework.train_model(model_type="xgboost")
        >>> 
        >>> # Тайлбарууд үүсгэх
        >>> explanations = framework.explain()
        >>> 
        >>> # Визуализаци
        >>> framework.visualize(explanation_type="summary")
        >>> 
        >>> # Dashboard ажиллуулах
        >>> framework.launch_dashboard()
    
    Атрибутууд:
        config (ConfigManager): Тохиргооны менежер instance
        data_processor: Өгөгдөл урьдчилан боловсруулах модуль
        model_trainer: Загвар сургах модуль
        explainer: SHAP тайлбарын модуль
        visualizer: Визуализацийн модуль
        evaluator: Хариуцлагатай AI үнэлгээний модуль
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        XAI Framework эхлүүлэх.
        
        Параметрүүд:
            config_path: Тусгай тохиргооны файлын зам (заавал биш)
        """
        self.config = ConfigManager()
        
        if config_path:
            self.config._config_path = Path(config_path)
            self.config.reload()
        
        # Бүрэлдэхүүн хэсгүүдийн placeholder-ууд (lazy loading)
        self._data_processor = None
        self._model_trainer = None
        self._explainer = None
        self._visualizer = None
        self._evaluator = None
        
        # Өгөгдлийн төлөв
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.model = None
        self.shap_values = None
        self.processing_info = None
        
        self._setup_logging()
        logger.info("XAIFramework эхэлсэн")
    
    def _setup_logging(self) -> None:
        """Тохиргооноос хамаарч logging тохируулах."""
        log_level = self.config.get("app.log_level", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # ==================== Өгөгдөл Боловсруулалт ====================
    
    @property
    def data_processor(self):
        """Data processor-ийг lazy load хийх."""
        if self._data_processor is None:
            from src.data_processing.processor import DataProcessor
            self._data_processor = DataProcessor(self.config)
        return self._data_processor
    
    def load_data(
        self,
        data_path: Union[str, Path, pd.DataFrame] = None,
        target_column: str = None,
        test_size: float = 0.2,
        # Нийцтэй байдлын өөр параметр нэрүүд
        data: Union[str, Path, pd.DataFrame] = None,
        target: str = None,
        protected_attributes: Optional[List[str]] = None
    ) -> 'XAIFramework':
        """
        Шинжилгээнд зориулж өгөгдөл ачаалж урьдчилан боловсруулах.
        
        Энэ метод:
        1. Файл эсвэл DataFrame-аас өгөгдөл ачаална
        2. Дутуу утгуудыг зохицуулна
        3. Категорийн хувьсагчдыг кодлоно
        4. Шинж чанаруудыг нормчилно
        5. Өгөгдөлд болзошгүй хазайлтыг илрүүлнэ
        6. Сургалт/тест багцуудад хуваана
        
        Параметрүүд:
            data_path: CSV файлын зам эсвэл pandas DataFrame
            target_column: Зорилтот баганын нэр
            test_size: Тестийн өгөгдлийн хувь (анхдагч 0.2)
            protected_attributes: Шударга байдлын шинжилгээнд хамгаалагдсан атрибутуудын жагсаалт
            
        Буцаах:
            self: Method chaining-д зориулсан
            
        Жишээ:
            >>> framework.load_data("data.csv", target_column="label", test_size=0.2)
        """
        # Өөр параметр нэрүүдийг зохицуулах (хоёр хэв маягийг дэмжих)
        if data_path is None and data is not None:
            data_path = data
        if target_column is None and target is not None:
            target_column = target
        
        if data_path is None:
            raise ValueError("data_path шаардлагатай")
        if target_column is None:
            raise ValueError("target_column шаардлагатай")
        
        # Шинэ өгөгдөл ачаалахад хуучин загвар болон SHAP утгуудыг цэвэрлэх
        # Энэ нь feature тооны зөрүүтэй алдаанаас сэргийлнэ
        if self.model is not None:
            logger.info("Шинэ өгөгдөл ачаалж байна - хуучин загварыг цэвэрлэж байна")
            self.model = None
            self.shap_values = None
            self.explanations = {}
            self._explainer = None
            
        logger.info(f"Зорилттой өгөгдөл ачаалж байна: {target_column}")
        
        # Өгөгдөл ачаалах
        if isinstance(data_path, (str, Path)):
            df = pd.read_csv(data_path)
        else:
            df = data_path.copy()
        
        # Шударга байдлын шинжилгээнд хамгаалагдсан атрибутуудыг хадгалах
        self._protected_attributes = protected_attributes or \
            self.config.get("responsible_ai.protected_attributes", [])
        
        # test_size-ийг баталгаажуулж тохируулах
        if test_size is None or test_size <= 0:
            test_size = 0.2  # Анхдагч утга
        if test_size >= 1:
            test_size = 0.2  # Буруу бол анхдагчаар тохируулах
        
        # Processor-ийн test_size-ийг шууд шинэчлэх
        self.data_processor.test_size = test_size
        
        processed = self.data_processor.process(df, target_column)
        
        self.X_train = processed['X_train']
        self.X_test = processed['X_test']
        self.y_train = processed['y_train']
        self.y_test = processed['y_test']
        self.feature_names = processed['feature_names']
        self.processing_info = processed.get('processing_info', {})
        self._original_data = df
        
        # Хамгаалагдсан атрибутуудын test өгөгдлийг хадгалах (fairness шинжилгээнд)
        self._protected_test_data = None
        if self._protected_attributes:
            available = [a for a in self._protected_attributes if a in df.columns]
            if available:
                # Processor-ийн train_test_split индексүүдийг ашиглах
                n_test = len(self.X_test)
                n_total = len(df)
                # Protected data-г feature_names-аас олох эсвэл original data-аас авах
                if hasattr(self.data_processor, '_test_indices'):
                    idx = self.data_processor._test_indices
                    self._protected_test_data = df[available].iloc[idx].reset_index(drop=True)
                else:
                    # Fallback: protected attr feature-ийн нэрэнд байвал X_test-ээс авах
                    attr_in_features = [a for a in available if a in self.feature_names]
                    if attr_in_features:
                        import pandas as _pd
                        feat_indices = [self.feature_names.index(a) for a in attr_in_features]
                        self._protected_test_data = _pd.DataFrame(
                            self.X_test[:, feat_indices],
                            columns=attr_in_features
                        )
        
        logger.info(f"Өгөгдөл ачаалагдлаа: {len(self.X_train)} сургалт, {len(self.X_test)} тест дээжүүд")
        logger.info(f"Шинж чанарууд: {len(self.feature_names)}")
        
        # Хазайлт илрүүлэлт
        if self.config.get("responsible_ai.bias_detection", True):
            self._detect_bias()
        
        return self
    
    def _detect_bias(self) -> None:
        """Өгөгдлийн багцад болзошгүй хазайлтыг илрүүлэх."""
        if not self._protected_attributes:
            return
            
        from src.evaluation.fairness import BiasDetector
        detector = BiasDetector()
        bias_report = detector.detect(
            self._original_data,
            self._protected_attributes
        )
        
        if bias_report['has_bias']:
            logger.warning(f"Болзошгүй хазайлт илэрлээ: {bias_report['summary']}")
        
        self._bias_report = bias_report
    
    # ==================== Загвар Сургалт ====================
    
    @property
    def model_trainer(self):
        """Model trainer-ийг lazy load хийх."""
        if self._model_trainer is None:
            from src.models.trainer import ModelTrainer
            self._model_trainer = ModelTrainer(self.config)
        return self._model_trainer
    
    def train_model(
        self,
        model_type: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'XAIFramework':
        """
        Ачаалсан өгөгдөл дээр хар хайрцагт загвар сургах.
        
        Дэмжигдсэн загварууд:
        - 'xgboost': XGBoost Classifier/Regressor
        - 'random_forest': Random Forest
        - 'neural_network': Deep Neural Network
        - 'lightgbm': LightGBM
        
        Параметрүүд:
            model_type: Сургах загварын төрөл (тохиргооноос анхдагч)
            model_params: Загварын hyperparameter-уудын dictionary
            **kwargs: Нэмэлт загварын hyperparameter-ууд
            
        Буцаах:
            self: Method chaining-д зориулсан
            
        Жишээ:
            >>> framework.train_model(model_type="xgboost", model_params={"n_estimators": 200})
        """
        if self.X_train is None:
            raise ValueError("Өгөгдөл ачаалаагүй байна. Эхлээд load_data() дуудна уу.")
        
        model_type = model_type or self.config.get("models.default_model", "xgboost")
        
        # model_params болон kwargs-ийг нэгтгэх
        params = {}
        if model_params:
            params.update(model_params)
        params.update(kwargs)
        
        logger.info(f"{model_type} загвар сургаж байна...")
        
        self.model = self.model_trainer.train(
            model_type=model_type,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
            **params
        )
        
        logger.info(f"Загварын сургалт дууслаа: {type(self.model).__name__}")
        
        return self
    
    # ==================== SHAP Тайлбарууд ====================
    
    @property
    def explainer(self):
        """SHAP explainer-ийг lazy load хийх."""
        if self._explainer is None:
            from src.explainers.shap_explainer import SHAPExplainer
            self._explainer = SHAPExplainer(self.config)
        return self._explainer
    
    def explain(
        self,
        explanation_type: Optional[str] = None,
        sample_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Загварын таамаглалуудад SHAP тайлбарууд үүсгэх.
        
        Энэ метод тооцоолно:
        - Глобал шинж чанарын ач холбогдол (бүх дээжүүд)
        - Локал тайлбарууд (тусдаа таамаглалууд)
        - Шинж чанарын харилцан үйлчлэлийн нөлөөлөл
        
        Параметрүүд:
            explanation_type: 'global', 'local', эсвэл 'both' (тохиргооноос анхдагч)
            sample_indices: Локал тайлбарт зориулсан тодорхой дээжүүд
            
        Буцаах:
            SHAP утгууд болон тайлбаруудыг агуулсан dictionary
            
        Жишээ:
            >>> explanations = framework.explain(explanation_type="both")
            >>> print(explanations['global']['feature_importance'])
        """
        if self.model is None:
            raise ValueError("Загвар сургаагүй байна. Эхлээд train_model() дуудна уу.")
        
        explanation_type = explanation_type or \
            self.config.get("shap.explanation_type", "both")
        
        logger.info(f"{explanation_type} тайлбарууд үүсгэж байна...")
        
        explanations = self.explainer.explain(
            model=self.model,
            X_background=self.X_train,
            X_explain=self.X_test,
            feature_names=self.feature_names,
            explanation_type=explanation_type,
            sample_indices=sample_indices
        )
        
        self.shap_values = explanations['shap_values']
        self._explanations = explanations
        
        logger.info("Тайлбарууд амжилттай үүсгэгдлээ")
        
        return explanations
    
    # ==================== Визуализаци ====================
    
    @property
    def visualizer(self):
        """Visualizer-ийг lazy load хийх."""
        if self._visualizer is None:
            from src.visualization.plots import XAIVisualizer
            self._visualizer = XAIVisualizer(self.config)
        return self._visualizer
    
    def visualize(
        self,
        plot_type: str = "summary",
        **kwargs
    ) -> Any:
        """
        SHAP тайлбаруудад визуализаци үүсгэх.
        
        Боломжит график төрлүүд:
        - 'summary': SHAP summary график (beeswarm)
        - 'bar': Шинж чанарын ач холбогдлын баар диаграм
        - 'waterfall': Нэг таамаглалд waterfall график
        - 'force': Таамаглалын тайлбарт force график
        - 'dependence': Шинж чанарын хамаарлын график
        - 'interaction': Шинж чанарын харилцан үйлчлэлийн heatmap
        
        Параметрүүд:
            plot_type: Визуализацийн төрөл
            **kwargs: Нэмэлт графикийн параметрүүд
            
        Буцаах:
            Plotly figure эсвэл matplotlib axes
            
        Жишээ:
            >>> fig = framework.visualize("summary")
            >>> fig.show()
        """
        if self.shap_values is None:
            raise ValueError("Тайлбарууд байхгүй байна. Эхлээд explain() дуудна уу.")
        
        return self.visualizer.plot(
            plot_type=plot_type,
            shap_values=self.shap_values,
            X=self.X_test,
            feature_names=self.feature_names,
            **kwargs
        )
    
    def launch_dashboard(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None
    ) -> None:
        """
        Интерактив визуал аналитик dashboard ажиллуулах.
        
        Dashboard хангана:
        - Загварын гүйцэтгэлийн хэмжигдэхүүнүүд
        - Интерактив SHAP визуализациуд
        - Тусдаа таамаглалын судлаач
        - Шударга байдлын шинжилгээ (идэвхжүүлсэн бол)
        
        Параметрүүд:
            host: Dashboard host (тохиргооноос анхдагч)
            port: Dashboard port (тохиргооноос анхдагч)
        """
        host = host or self.config.get("dashboard.host", "localhost")
        port = port or self.config.get("dashboard.port", 8501)
        
        logger.info(f"Dashboard ажиллуулж байна http://{host}:{port}")
        
        from src.dashboard.app import run_dashboard
        run_dashboard(
            framework=self,
            host=host,
            port=port
        )
    
    # ==================== Үнэлгээ ====================
    
    @property
    def evaluator(self):
        """Evaluator-ийг lazy load хийх."""
        if self._evaluator is None:
            from src.evaluation.metrics import ModelEvaluator
            self._evaluator = ModelEvaluator(self.config)
        return self._evaluator
    
    def evaluate(
        self,
        include_fairness: bool = True,
        decision_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Хариуцлагатай AI хэмжигдэхүүнүүдийг оролцуулсан иж бүрэн загварын үнэлгээ.
        
        Үнэлдэг:
        - Загварын гүйцэтгэл (accuracy, precision, recall, F1, AUC)
        - Тайлбарын чанар (тогтвортой байдал, тогтмол байдал)
        - Шударга байдлын хэмжигдэхүүнүүд (хамгаалагдсан атрибутууд заасан бол)
        - Ил тод байдлын хэмжүүрүүд
        
        Параметрүүд:
            include_fairness: Шударга байдлын үнэлгээг оруулах эсэх
            
        Буцаах:
            Бүх үнэлгээний хэмжигдэхүүнүүдийг агуулсан dictionary
        """
        if self.model is None:
            raise ValueError("Үнэлэх загвар байхгүй. Эхлээд train_model() дуудна уу.")
        
        results = self.evaluator.evaluate(
            model=self.model,
            X_test=self.X_test,
            y_test=self.y_test,
            shap_values=self.shap_values,
            decision_threshold=decision_threshold,
        )
        
        if include_fairness and self._protected_attributes:
            from src.evaluation.fairness import FairnessEvaluator
            fairness_eval = FairnessEvaluator()
            
            # Хамгаалагдсан атрибутуудын өгөгдлийг авах
            protected_data = getattr(self, '_protected_test_data', None)
            
            results['fairness'] = fairness_eval.evaluate(
                model=self.model,
                X_test=self.X_test,
                y_test=self.y_test,
                protected_attributes=self._protected_attributes,
                protected_data=protected_data,
                decision_threshold=decision_threshold,
            )
        
        self._evaluation_results = results
        
        return results
    
    # ==================== Экспорт & Тайлан ====================
    
    def export_report(
        self,
        output_path: Union[str, Path],
        format: str = "html"
    ) -> None:
        """
        Иж бүрэн шинжилгээний тайлан экспортлох.
        
        Параметрүүд:
            output_path: Тайлан хадгалах зам
            format: Экспортын формат ('html', 'pdf', 'json')
        """
        from src.utils.reporting import ReportGenerator
        
        generator = ReportGenerator()
        generator.generate(
            framework=self,
            output_path=output_path,
            format=format
        )
        
        logger.info(f"Тайлан {output_path} руу экспортлогдлоо")
    
    def save_state(self, path: Union[str, Path]) -> None:
        """Framework-ийн төлөвийг дараа ашиглахаар хадгалах."""
        import joblib
        
        state = {
            'model': self.model,
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'feature_names': self.feature_names,
            'shap_values': self.shap_values,
            'config': self.config.all
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(state, path)
        logger.info(f"Framework-ийн төлөв {path} руу хадгалагдлаа")
    
    @classmethod
    def load_state(cls, path: Union[str, Path]) -> 'XAIFramework':
        """Хадгалсан төлөвөөс framework ачаалах."""
        import joblib
        
        state = joblib.load(path)
        
        framework = cls()
        framework.model = state['model']
        framework.X_train = state['X_train']
        framework.X_test = state['X_test']
        framework.y_train = state['y_train']
        framework.y_test = state['y_test']
        framework.feature_names = state['feature_names']
        framework.shap_values = state['shap_values']
        
        logger.info(f"Framework-ийн төлөв {path}-аас ачаалагдлаа")
        
        return framework
    
    def __repr__(self) -> str:
        model_info = type(self.model).__name__ if self.model else "Байхгүй"
        data_info = f"{len(self.X_train)} дээж" if self.X_train is not None else "Өгөгдөл байхгүй"
        return f"XAIFramework(загвар={model_info}, өгөгдөл={data_info})"
