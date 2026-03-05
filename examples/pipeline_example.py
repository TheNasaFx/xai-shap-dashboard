"""
Complete XAI Pipeline Example
==============================

Demonstrates the end-to-end automated pipeline functionality.

Author: XAI-SHAP Framework
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.core.pipeline import XAIPipeline

def main():
    """
    Run the complete XAI pipeline.
    
    The pipeline automates:
    1. Data loading and preprocessing
    2. Model training
    3. SHAP explanation generation
    4. Visualization creation
    5. Model evaluation
    """
    
    print("=" * 60)
    print("XAI-SHAP Visual Analytics Framework - Pipeline Example")
    print("=" * 60)
    
    # Create sample data for demonstration
    from sklearn.datasets import load_breast_cancer
    import pandas as pd
    
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['diagnosis'] = data.target  # 0 = malignant, 1 = benign
    
    # Save as CSV for pipeline
    df.to_csv("data/sample/breast_cancer.csv", index=False)
    print("✅ Sample data saved to data/sample/breast_cancer.csv")
    
    # =========================================
    # Option 1: Run Pipeline with Configuration
    # =========================================
    print("\n📦 Running XAI Pipeline...")
    
    # Create pipeline configuration
    config = {
        'data': {
            'path': 'data/sample/breast_cancer.csv',
            'target_column': 'diagnosis',
            'test_size': 0.2
        },
        'model': {
            'type': 'xgboost',
            'params': {
                'n_estimators': 100,
                'max_depth': 5,
                'random_state': 42
            }
        },
        'explanation': {
            'n_samples': 100
        },
        'output': {
            'save_plots': True,
            'output_dir': 'outputs/pipeline_results'
        }
    }
    
    # Create and run pipeline
    pipeline = XAIPipeline(config=config)
    
    # Run all stages
    pipeline.run()
    
    # Get results
    results = pipeline.get_results()
    
    print("\n" + "=" * 60)
    print("📈 Pipeline Results")
    print("=" * 60)
    
    if 'evaluation' in results and 'accuracy' in results['evaluation']:
        print(f"   Accuracy:  {results['evaluation']['accuracy']:.4f}")
        print(f"   F1 Score:  {results['evaluation'].get('f1_score', 'N/A')}")
    
    if 'feature_importance' in results:
        print("\n   Top 5 Important Features:")
        top_features = sorted(
            results['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for i, (feature, importance) in enumerate(top_features, 1):
            print(f"   {i}. {feature}: {importance:.4f}")
    
    # =========================================
    # Option 2: Step-by-Step Framework Usage
    # =========================================
    print("\n" + "=" * 60)
    print("📊 Step-by-Step Framework Execution")
    print("=" * 60)
    
    from src.core.framework import XAIFramework
    
    framework = XAIFramework()
    
    # Load data
    print("\n   Step 1: Loading data...")
    framework.load_data(
        data_path='data/sample/breast_cancer.csv',
        target_column='diagnosis'
    )
    
    # Train model
    print("   Step 2: Training model...")
    framework.train_model(model_type='random_forest', model_params={'n_estimators': 50})
    
    # Explain model
    print("   Step 3: Generating explanations...")
    framework.explain(explanation_type='both')
    
    # Create visualizations
    print("   Step 4: Creating visualizations...")
    import os
    os.makedirs('outputs/pipeline_step_by_step', exist_ok=True)
    
    for plot_type in ['summary', 'bar']:
        fig = framework.visualize(plot_type=plot_type)
        fig.write_html(f'outputs/pipeline_step_by_step/{plot_type}_plot.html')
    
    # Evaluate
    print("   Step 5: Evaluating model...")
    from sklearn.metrics import accuracy_score
    y_pred = framework.model.predict(framework.X_test)
    accuracy = accuracy_score(framework.y_test, y_pred)
    
    print(f"\n   Results: Accuracy = {accuracy:.4f}")
    
    print("\n✨ Pipeline example complete!")
    print("   Check 'outputs/' directory for visualizations and reports.")


if __name__ == "__main__":
    main()
