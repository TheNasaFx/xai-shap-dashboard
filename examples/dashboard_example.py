"""
Dashboard Usage Example
========================

Demonstrates how to launch and configure the interactive Streamlit dashboard.

Author: XAI-SHAP Framework
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """
    Launch the XAI-SHAP Interactive Dashboard.
    
    Run this script to start the Streamlit dashboard:
        python examples/dashboard_example.py
    
    Or run directly with Streamlit:
        streamlit run src/dashboard/app.py
    """
    
    print("=" * 60)
    print("XAI-SHAP Visual Analytics Framework - Interactive Dashboard")
    print("=" * 60)
    print("""
The dashboard provides an interactive interface for:

📊 Data Exploration
   - Upload your own CSV data
   - View data statistics and distributions
   - Detect missing values and outliers

🤖 Model Training
   - Train XGBoost, Random Forest, or Neural Network models
   - Configure hyperparameters
   - View training progress

🔍 SHAP Explanations
   - Generate local and global explanations
   - Interactive feature importance analysis
   - Individual prediction breakdowns

📈 Visualizations
   - Summary plots
   - Waterfall plots
   - Force plots
   - Dependence plots

⚖️ Fairness Analysis
   - Demographic parity
   - Equalized odds
   - Disparate impact metrics

To start the dashboard, run one of the following commands:
    """)
    
    print("1. Using Python:")
    print("   python -c \"from src.dashboard.app import run_dashboard; run_dashboard()\"")
    
    print("\n2. Using Streamlit directly:")
    print("   streamlit run src/dashboard/app.py")
    
    print("\n3. Programmatic setup with pre-loaded data:")
    
    # Example of programmatic setup
    example_code = '''
    from src.core.framework import XAIFramework
    from src.dashboard.app import create_dashboard
    
    # Prepare framework with data
    framework = XAIFramework()
    framework.load_data(
        data_path="data/your_data.csv",
        target_column="target"
    )
    framework.train_model(model_type="xgboost")
    framework.explain_model()
    
    # Create dashboard with pre-loaded framework
    # (Note: This is conceptual - dashboard runs via Streamlit)
    '''
    
    print(example_code)
    
    # Ask if user wants to launch dashboard
    print("\n" + "-" * 60)
    response = input("Would you like to launch the dashboard now? (y/n): ")
    
    if response.lower() == 'y':
        print("\n🚀 Launching dashboard...")
        print("   Opening in browser at http://localhost:8501")
        print("   Press Ctrl+C to stop the server.\n")
        
        import subprocess
        import sys
        
        try:
            subprocess.run([
                sys.executable, "-m", "streamlit", "run",
                "src/dashboard/app.py",
                "--server.port=8501"
            ])
        except KeyboardInterrupt:
            print("\n\n👋 Dashboard stopped.")
    else:
        print("\n📝 To launch later, run: streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    main()
