"""
XAI Pipeline - Automated Analysis Pipeline
============================================

Provides automated pipeline for running complete XAI analysis
from data loading to report generation.

Author: XAI-SHAP Framework
"""

import logging
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
import pandas as pd

from src.core.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class XAIPipeline:
    """
    Automated pipeline for complete XAI analysis workflow.
    
    This class provides a streamlined interface for running
    end-to-end explainable AI analysis with minimal configuration.
    
    Pipeline stages:
    1. Data Loading & Preprocessing
    2. Bias Detection
    3. Model Training
    4. SHAP Explanation Generation
    5. Visualization Creation
    6. Fairness Evaluation
    7. Report Generation
    
    Example:
        >>> from src.core.pipeline import XAIPipeline
        >>> 
        >>> # Quick analysis
        >>> results = XAIPipeline.quick_analysis(
        ...     data="data/sample.csv",
        ...     target="label",
        ...     model_type="xgboost"
        ... )
        >>> 
        >>> # Custom pipeline
        >>> pipeline = XAIPipeline(config)
        >>> pipeline.add_stage("load_data", params={"path": "data.csv"})
        >>> pipeline.add_stage("train_model", params={"model": "xgboost"})
        >>> pipeline.add_stage("explain")
        >>> pipeline.add_stage("visualize")
        >>> results = pipeline.run()
    """
    
    def __init__(self, config: Optional[Union[ConfigManager, Dict]] = None):
        """
        Initialize the XAI Pipeline.
        
        Args:
            config: Configuration manager or dict (creates new if not provided)
        """
        if isinstance(config, dict):
            self._config_dict = config
            self.config = ConfigManager()
        else:
            self._config_dict = {}
            self.config = config or ConfigManager()
        self.stages: List[Dict[str, Any]] = []
        self._results: Dict[str, Any] = {}
        self._framework = None
        
        # Auto-setup stages from config dict
        if self._config_dict:
            self._setup_from_config()
        
        logger.info("XAIPipeline initialized")
    
    def _setup_from_config(self):
        """Set up pipeline stages from config dict."""
        cfg = self._config_dict
        
        # Add data loading stage
        if 'data' in cfg:
            self.add_stage('load_data', params={
                'path': cfg['data'].get('path'),
                'target': cfg['data'].get('target_column', 'target'),
            })
        
        # Add model training stage
        if 'model' in cfg:
            self.add_stage('train_model', params={
                'model_type': cfg['model'].get('type', 'xgboost'),
                **cfg['model'].get('params', {})
            })
        
        # Add explanation stage
        if 'explanation' in cfg:
            self.add_stage('explain', params=cfg['explanation'])
        else:
            self.add_stage('explain')
        
        # Add visualization stage
        if 'output' in cfg and cfg['output'].get('save_plots'):
            self.add_stage('visualize', params={
                'save_path': cfg['output'].get('output_dir')
            })
    
    @property
    def framework(self):
        """Get or create framework instance."""
        if self._framework is None:
            from src.core.framework import XAIFramework
            self._framework = XAIFramework()
        return self._framework
    
    def add_stage(
        self,
        stage_name: str,
        params: Optional[Dict[str, Any]] = None,
        condition: Optional[callable] = None
    ) -> 'XAIPipeline':
        """
        Add a stage to the pipeline.
        
        Available stages:
        - 'load_data': Load and preprocess data
        - 'train_model': Train ML model
        - 'explain': Generate SHAP explanations
        - 'visualize': Create visualizations
        - 'evaluate': Run model evaluation
        - 'fairness': Evaluate fairness metrics
        - 'export': Export report
        
        Args:
            stage_name: Name of the pipeline stage
            params: Parameters for the stage
            condition: Optional condition function
            
        Returns:
            self: For method chaining
        """
        self.stages.append({
            'name': stage_name,
            'params': params or {},
            'condition': condition,
            'status': 'pending'
        })
        
        return self
    
    def run(self) -> Dict[str, Any]:
        """
        Execute all pipeline stages in order.
        
        Returns:
            Dictionary containing results from all stages
        """
        logger.info(f"Starting pipeline with {len(self.stages)} stages")
        
        for i, stage in enumerate(self.stages):
            stage_name = stage['name']
            params = stage['params']
            condition = stage['condition']
            
            # Check condition
            if condition and not condition(self._results):
                logger.info(f"Skipping stage '{stage_name}' (condition not met)")
                stage['status'] = 'skipped'
                continue
            
            logger.info(f"Running stage {i+1}/{len(self.stages)}: {stage_name}")
            
            try:
                result = self._execute_stage(stage_name, params)
                self._results[stage_name] = result
                stage['status'] = 'completed'
            except Exception as e:
                logger.error(f"Stage '{stage_name}' failed: {e}")
                stage['status'] = 'failed'
                stage['error'] = str(e)
                raise
        
        logger.info("Pipeline completed successfully")
        return self._results
    
    def _execute_stage(self, stage_name: str, params: Dict[str, Any]) -> Any:
        """Execute a single pipeline stage."""
        stage_methods = {
            'load_data': self._stage_load_data,
            'train_model': self._stage_train_model,
            'explain': self._stage_explain,
            'visualize': self._stage_visualize,
            'evaluate': self._stage_evaluate,
            'fairness': self._stage_fairness,
            'export': self._stage_export,
        }
        
        if stage_name not in stage_methods:
            raise ValueError(f"Unknown stage: {stage_name}")
        
        return stage_methods[stage_name](**params)
    
    def _stage_load_data(
        self,
        path: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        target: str = "target",
        protected_attributes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Load data stage."""
        self.framework.load_data(
            data=path or data,
            target=target,
            protected_attributes=protected_attributes
        )
        
        return {
            'n_train': len(self.framework.X_train),
            'n_test': len(self.framework.X_test),
            'n_features': len(self.framework.feature_names),
            'features': self.framework.feature_names
        }
    
    def _stage_train_model(
        self,
        model_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train model stage."""
        self.framework.train_model(model_type=model_type, **kwargs)
        
        return {
            'model_type': type(self.framework.model).__name__,
            'model': self.framework.model
        }
    
    def _stage_explain(
        self,
        explanation_type: str = "both",
        sample_indices: Optional[List[int]] = None,
        n_samples: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate explanations stage."""
        explanations = self.framework.explain(
            explanation_type=explanation_type,
            sample_indices=sample_indices
        )
        
        return explanations
    
    def _stage_visualize(
        self,
        plot_types: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create visualizations stage."""
        plot_types = plot_types or ['summary', 'bar', 'waterfall']
        figures = {}
        
        # Create output directory if needed
        if save_path:
            Path(save_path).mkdir(parents=True, exist_ok=True)
        
        for plot_type in plot_types:
            fig = self.framework.visualize(plot_type=plot_type)
            figures[plot_type] = fig
            
            if save_path:
                fig_path = Path(save_path) / f"{plot_type}_plot.html"
                fig.write_html(str(fig_path))
        
        return {'figures': figures}
    
    def _stage_evaluate(self, include_fairness: bool = False) -> Dict[str, Any]:
        """Evaluate model stage."""
        return self.framework.evaluate(include_fairness=include_fairness)
    
    def _stage_fairness(
        self,
        protected_attributes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Fairness evaluation stage."""
        from src.evaluation.fairness import FairnessEvaluator
        
        evaluator = FairnessEvaluator()
        return evaluator.evaluate(
            model=self.framework.model,
            X_test=self.framework.X_test,
            y_test=self.framework.y_test,
            protected_attributes=protected_attributes or []
        )
    
    def _stage_export(
        self,
        output_path: str,
        format: str = "html"
    ) -> Dict[str, Any]:
        """Export report stage."""
        self.framework.export_report(output_path=output_path, format=format)
        return {'path': output_path, 'format': format}
    
    def status(self) -> Dict[str, str]:
        """Get status of all pipeline stages."""
        return {stage['name']: stage['status'] for stage in self.stages}
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get results from all completed pipeline stages.
        
        Returns:
            Dictionary containing results from all completed stages
        """
        return self._results
    
    def reset(self) -> 'XAIPipeline':
        """Reset pipeline for re-execution."""
        for stage in self.stages:
            stage['status'] = 'pending'
        self._results = {}
        return self
    
    def clear(self) -> 'XAIPipeline':
        """Clear all stages from pipeline."""
        self.stages = []
        self._results = {}
        return self
    
    @classmethod
    def quick_analysis(
        cls,
        data: Union[str, pd.DataFrame],
        target: str,
        model_type: str = "xgboost",
        protected_attributes: Optional[List[str]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run quick end-to-end XAI analysis.
        
        This is a convenience method for running complete analysis
        with default settings.
        
        Args:
            data: Path to CSV or DataFrame
            target: Target column name
            model_type: Type of model to train
            protected_attributes: Protected attributes for fairness
            output_path: Path to save report
            
        Returns:
            Dictionary containing all analysis results
            
        Example:
            >>> results = XAIPipeline.quick_analysis(
            ...     data="data/sample.csv",
            ...     target="label"
            ... )
            >>> print(results['evaluate']['accuracy'])
        """
        pipeline = cls()
        
        # Build pipeline
        pipeline.add_stage('load_data', {
            'path' if isinstance(data, str) else 'data': data,
            'target': target,
            'protected_attributes': protected_attributes
        })
        
        pipeline.add_stage('train_model', {
            'model_type': model_type
        })
        
        pipeline.add_stage('explain')
        pipeline.add_stage('evaluate', {'include_fairness': bool(protected_attributes)})
        
        if output_path:
            pipeline.add_stage('export', {'output_path': output_path})
        
        return pipeline.run()
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'XAIPipeline':
        """
        Create pipeline from YAML configuration.
        
        Args:
            yaml_path: Path to YAML pipeline configuration
            
        Returns:
            Configured XAIPipeline instance
        """
        import yaml
        
        with open(yaml_path, 'r') as f:
            pipeline_config = yaml.safe_load(f)
        
        pipeline = cls()
        
        for stage in pipeline_config.get('stages', []):
            pipeline.add_stage(
                stage_name=stage['name'],
                params=stage.get('params', {})
            )
        
        return pipeline
    
    def __repr__(self) -> str:
        status = self.status()
        completed = sum(1 for s in status.values() if s == 'completed')
        return f"XAIPipeline(stages={len(self.stages)}, completed={completed})"
