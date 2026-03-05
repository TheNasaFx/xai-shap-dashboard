"""
Report Generator - Comprehensive Analysis Reports
===================================================

Generates comprehensive reports from XAI analysis results.

Author: XAI-SHAP Framework
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
import json

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate comprehensive analysis reports.
    
    Supports multiple output formats:
    - HTML: Interactive web report
    - PDF: Printable document
    - JSON: Machine-readable data
    - Markdown: Documentation-friendly
    
    Example:
        >>> generator = ReportGenerator()
        >>> generator.generate(framework, "report.html", format="html")
    """
    
    def __init__(self):
        """Initialize ReportGenerator."""
        self._template_html = self._get_html_template()
    
    def generate(
        self,
        framework,
        output_path: Union[str, Path],
        format: str = "html",
        title: str = "XAI Analysis Report"
    ) -> None:
        """
        Generate report from framework analysis.
        
        Args:
            framework: XAIFramework instance with completed analysis
            output_path: Path to save report
            format: Output format ('html', 'pdf', 'json', 'markdown')
            title: Report title
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Collect report data
        report_data = self._collect_report_data(framework, title)
        
        # Generate report in specified format
        if format == "html":
            content = self._generate_html(report_data)
        elif format == "json":
            content = json.dumps(report_data, indent=2, default=str)
        elif format == "markdown":
            content = self._generate_markdown(report_data)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Report generated: {output_path}")
    
    def _collect_report_data(self, framework, title: str) -> Dict[str, Any]:
        """Collect all data for report."""
        return {
            'title': title,
            'generated_at': datetime.now().isoformat(),
            'framework_version': '1.0.0',
            'data_summary': self._get_data_summary(framework),
            'model_info': self._get_model_info(framework),
            'explanation_summary': self._get_explanation_summary(framework),
            'evaluation_results': getattr(framework, '_evaluation_results', {})
        }
    
    def _get_data_summary(self, framework) -> Dict[str, Any]:
        """Get data summary."""
        if framework.X_train is None:
            return {'status': 'No data loaded'}
        
        return {
            'n_train_samples': len(framework.X_train),
            'n_test_samples': len(framework.X_test),
            'n_features': len(framework.feature_names),
            'feature_names': framework.feature_names
        }
    
    def _get_model_info(self, framework) -> Dict[str, Any]:
        """Get model information."""
        if framework.model is None:
            return {'status': 'No model trained'}
        
        return {
            'model_type': type(framework.model).__name__,
            'model_class': str(type(framework.model))
        }
    
    def _get_explanation_summary(self, framework) -> Dict[str, Any]:
        """Get explanation summary."""
        if framework.shap_values is None:
            return {'status': 'No explanations generated'}
        
        import numpy as np
        
        # Feature importance
        mean_abs_shap = np.abs(framework.shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:10]
        
        return {
            'status': 'Generated',
            'n_samples_explained': len(framework.shap_values),
            'top_features': [
                {
                    'feature': framework.feature_names[i],
                    'importance': float(mean_abs_shap[i])
                }
                for i in top_indices
            ]
        }
    
    def _generate_html(self, data: Dict[str, Any]) -> str:
        """Generate HTML report."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{data['title']}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            border-radius: 8px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .meta {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #3498db;
            color: white;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .metric {{
            display: inline-block;
            padding: 15px 25px;
            margin: 5px;
            background: #ecf0f1;
            border-radius: 8px;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2980b9;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .success {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .error {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 {data['title']}</h1>
        <p class="meta">Generated: {data['generated_at']} | Framework v{data['framework_version']}</p>
    </div>
    
    <div class="container">
        <h2>📊 Data Summary</h2>
        {self._format_data_summary_html(data['data_summary'])}
    </div>
    
    <div class="container">
        <h2>🤖 Model Information</h2>
        {self._format_model_info_html(data['model_info'])}
    </div>
    
    <div class="container">
        <h2>🔍 Explanation Summary</h2>
        {self._format_explanation_html(data['explanation_summary'])}
    </div>
    
    <div class="container">
        <h2>📈 Evaluation Results</h2>
        {self._format_evaluation_html(data['evaluation_results'])}
    </div>
    
    <footer style="text-align: center; color: #7f8c8d; padding: 20px;">
        Generated by XAI-SHAP Visual Analytics Framework
    </footer>
</body>
</html>
        """
        return html
    
    def _format_data_summary_html(self, summary: Dict) -> str:
        """Format data summary as HTML."""
        if summary.get('status') == 'No data loaded':
            return '<p class="warning">No data has been loaded.</p>'
        
        return f"""
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{summary.get('n_train_samples', 'N/A')}</div>
                <div class="metric-label">Training Samples</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary.get('n_test_samples', 'N/A')}</div>
                <div class="metric-label">Test Samples</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary.get('n_features', 'N/A')}</div>
                <div class="metric-label">Features</div>
            </div>
        </div>
        """
    
    def _format_model_info_html(self, info: Dict) -> str:
        """Format model info as HTML."""
        if info.get('status') == 'No model trained':
            return '<p class="warning">No model has been trained.</p>'
        
        return f"""
        <p><strong>Model Type:</strong> {info.get('model_type', 'Unknown')}</p>
        """
    
    def _format_explanation_html(self, summary: Dict) -> str:
        """Format explanation summary as HTML."""
        if summary.get('status') != 'Generated':
            return '<p class="warning">No explanations have been generated.</p>'
        
        features_html = "".join(
            f"<tr><td>{f['feature']}</td><td>{f['importance']:.4f}</td></tr>"
            for f in summary.get('top_features', [])
        )
        
        return f"""
        <p><strong>Samples Explained:</strong> {summary.get('n_samples_explained', 'N/A')}</p>
        <h3>Top 10 Important Features</h3>
        <table>
            <tr><th>Feature</th><th>Importance (Mean |SHAP|)</th></tr>
            {features_html}
        </table>
        """
    
    def _format_evaluation_html(self, results: Dict) -> str:
        """Format evaluation results as HTML."""
        if not results:
            return '<p class="warning">No evaluation has been performed.</p>'
        
        html_parts = []
        
        if 'classification' in results:
            metrics = results['classification']
            html_parts.append(f"""
            <h3>Classification Metrics</h3>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">{metrics.get('accuracy', 0):.2%}</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{metrics.get('f1_score', 0):.2%}</div>
                    <div class="metric-label">F1 Score</div>
                </div>
            </div>
            """)
        
        return "".join(html_parts) or '<p>Results available in JSON format.</p>'
    
    def _generate_markdown(self, data: Dict[str, Any]) -> str:
        """Generate Markdown report."""
        lines = [
            f"# {data['title']}",
            f"\n*Generated: {data['generated_at']}*\n",
            "## Data Summary\n"
        ]
        
        ds = data['data_summary']
        if ds.get('status') != 'No data loaded':
            lines.extend([
                f"- **Training Samples:** {ds.get('n_train_samples', 'N/A')}",
                f"- **Test Samples:** {ds.get('n_test_samples', 'N/A')}",
                f"- **Features:** {ds.get('n_features', 'N/A')}\n"
            ])
        
        lines.append("## Model Information\n")
        mi = data['model_info']
        if mi.get('status') != 'No model trained':
            lines.append(f"- **Model Type:** {mi.get('model_type', 'Unknown')}\n")
        
        lines.append("## Feature Importance\n")
        es = data['explanation_summary']
        if es.get('status') == 'Generated':
            lines.append("| Feature | Importance |")
            lines.append("|---------|------------|")
            for f in es.get('top_features', []):
                lines.append(f"| {f['feature']} | {f['importance']:.4f} |")
        
        return "\n".join(lines)
    
    def _get_html_template(self) -> str:
        """Get base HTML template."""
        return """<!DOCTYPE html><html><head></head><body></body></html>"""
