"""
XAI-SHAP Framework: Basic Usage Example
=========================================

This example demonstrates the core functionality of the XAI-SHAP 
Visual Analytics Framework for Explainable AI.

Author: XAI-SHAP Framework
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import warnings
warnings.filterwarnings('ignore')

# Import the framework
from src.core.framework import XAIFramework

def main():
    """
    Complete example of using the XAI-SHAP framework.
    
    This example covers:
    1. Data loading and preprocessing
    2. Model training (XGBoost, Random Forest, Neural Network)
    3. SHAP explanation generation
    4. Visualization of results
    5. Model evaluation with fairness metrics
    6. Report generation
    """
    
    print("=" * 60)
    print("XAI-SHAP Visual Analytics Framework - Basic Usage Example")
    print("=" * 60)
    
    # =========================================
    # Step 1: Initialize Framework
    # =========================================
    print("\n📦 Step 1: Initializing Framework...")
    
    # Initialize with optional config file
    # framework = XAIFramework(config_path="config/config.yaml")
    framework = XAIFramework()
    
    print("✅ Framework initialized successfully!")
    
    # =========================================
    # Step 2: Load and Prepare Data
    # =========================================
    print("\n📊 Step 2: Loading Data...")
    
    # Option A: Load from CSV file
    # framework.load_data(
    #     data_path="data/sample/dataset.csv",
    #     target_column="target",
    #     test_size=0.2
    # )
    
    # Option B: Use sample data (for demonstration)
    from sklearn.datasets import make_classification
    import pandas as pd
    
    # Generate sample classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    # Create DataFrame with feature names
    feature_names = [f"feature_{i}" for i in range(10)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Load data into framework
    framework.load_data(
        data_path=df,  # Can accept DataFrame directly
        target_column="target",
        test_size=0.2
    )
    
    print(f"✅ Data loaded: {len(framework.X_train)} training samples, {len(framework.X_test)} test samples")
    print(f"   Features: {framework.feature_names}")
    
    # =========================================
    # Step 3: Train Model
    # =========================================
    print("\n🤖 Step 3: Training Model...")
    
    # Train XGBoost model
    framework.train_model(
        model_type="xgboost",
        model_params={
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42
        }
    )
    
    print(f"✅ Model trained: {type(framework.model).__name__}")
    
    # Make predictions
    from sklearn.metrics import accuracy_score
    y_pred = framework.model.predict(framework.X_test)
    accuracy = accuracy_score(framework.y_test, y_pred)
    print(f"   Test Accuracy: {accuracy:.4f}")
    
    # =========================================
    # Step 4: Generate SHAP Explanations
    # =========================================
    print("\n🔍 Step 4: Generating SHAP Explanations...")
    
    explanations = framework.explain(
        explanation_type="both"
    )
    
    print(f"✅ SHAP values computed!")
    print(f"   Shape: {framework.shap_values.shape}")
    
    # Get feature importance
    import numpy as np
    mean_shap = np.abs(framework.shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_shap)[::-1]
    
    print("\n📈 Feature Importance (by SHAP):")
    for i in range(min(5, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f"   {i+1}. {framework.feature_names[idx]}: {mean_shap[idx]:.4f}")
    
    # =========================================
    # Step 5: Visualize Explanations
    # =========================================
    print("\n📊 Step 5: Creating Visualizations...")
    
    # Ensure outputs directory exists
    import os
    os.makedirs("outputs", exist_ok=True)
    
    # Create visualizer
    from src.visualization.plots import XAIVisualizer
    visualizer = XAIVisualizer()
    
    # Generate summary plot using plot method
    fig = visualizer.plot(
        plot_type="summary",
        shap_values=framework.shap_values,
        X=framework.X_test[:100] if hasattr(framework, 'X_test') else None,
        feature_names=framework.feature_names
    )
    if fig is not None:
        fig.write_html("outputs/summary_plot.html")
        print("   ✅ Summary plot saved to outputs/summary_plot.html")
    
    # Generate bar plot (feature importance)
    fig = visualizer.plot(
        plot_type="bar",
        shap_values=framework.shap_values,
        X=framework.X_test,
        feature_names=framework.feature_names
    )
    if fig is not None:
        fig.write_html("outputs/feature_importance.html")
        print("   ✅ Feature importance plot saved to outputs/feature_importance.html")
    
    # Generate waterfall plot for single instance
    fig = visualizer.plot(
        plot_type="waterfall",
        shap_values=framework.shap_values,
        feature_names=framework.feature_names,
        X=framework.X_test if hasattr(framework, 'X_test') else None,
        sample_idx=0
    )
    if fig is not None:
        fig.write_html("outputs/waterfall_plot.html")
        print("   ✅ Waterfall plot saved to outputs/waterfall_plot.html")
    
    # =========================================
    # Step 6: Evaluate Model
    # =========================================
    print("\n📈 Step 6: Evaluating Model...")
    
    from src.evaluation.metrics import ModelEvaluator
    
    evaluator = ModelEvaluator()
    
    # Classification evaluation
    eval_results = evaluator.evaluate_classification(
        y_true=framework.y_test,
        y_pred=y_pred
    )
    
    print("   Classification Metrics:")
    print(f"   - Accuracy:  {eval_results['accuracy']:.4f}")
    print(f"   - Precision: {eval_results['precision']:.4f}")
    print(f"   - Recall:    {eval_results['recall']:.4f}")
    print(f"   - F1 Score:  {eval_results['f1_score']:.4f}")
    
    # =========================================
    # Step 7: Generate Report
    # =========================================
    print("\n📝 Step 7: Generating Report...")
    
    from src.utils.reporting import ReportGenerator
    
    report_gen = ReportGenerator()
    
    # Generate HTML report
    report_gen.generate(
        framework=framework,
        output_path="outputs/analysis_report.html",
        format="html",
        title="XAI Analysis Report"
    )
    print("   ✅ HTML report saved to outputs/analysis_report.html")
    
    # Generate Markdown report
    report_gen.generate(
        framework=framework,
        output_path="outputs/analysis_report.md",
        format="markdown",
        title="XAI Analysis Report"
    )
    print("   ✅ Markdown report saved to outputs/analysis_report.md")
    
    # =========================================
    # Summary
    # =========================================
    print("\n" + "=" * 60)
    print("✨ Analysis Complete!")
    print("=" * 60)
    print("""
Output files:
  📊 outputs/summary_plot.html      - SHAP summary plot
  📊 outputs/feature_importance.html - Feature importance bar chart
  📊 outputs/waterfall_plot.html    - Individual prediction explanation
  📝 outputs/analysis_report.html   - Comprehensive HTML report
  📝 outputs/analysis_report.md     - Markdown report
    """)
    
    return framework


def example_with_real_data():
    """
    Example using real dataset (requires data file).
    """
    from src.core.framework import XAIFramework
    
    framework = XAIFramework()
    
    # Load your data
    framework.load_data(
        data_path="data/your_dataset.csv",
        target_column="your_target_column",
        test_size=0.2
    )
    
    # Train model
    framework.train_model(model_type="xgboost")
    
    # Explain
    framework.explain_model()
    
    # Visualize
    framework.visualize(plot_type="summary", save_path="outputs/my_analysis.html")
    
    return framework


def example_model_comparison():
    """
    Example comparing multiple models.
    """
    from src.core.framework import XAIFramework
    from sklearn.datasets import load_breast_cancer
    import pandas as pd
    
    # Load breast cancer dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    models = ['xgboost', 'random_forest']
    results = {}
    
    for model_type in models:
        print(f"\n{'=' * 40}")
        print(f"Training {model_type.upper()}")
        print('=' * 40)
        
        framework = XAIFramework()
        framework.load_data(data_path=df, target_column="target")
        framework.train_model(model_type=model_type)
        framework.explain_model()
        
        # Save results
        results[model_type] = {
            'framework': framework,
            'shap_values': framework.shap_values
        }
        
        # Save visualization
        framework.visualize(
            plot_type="summary",
            save_path=f"outputs/{model_type}_summary.html"
        )
    
    return results


if __name__ == "__main__":
    main()
