"""
Fairness Analysis Example
==========================

Demonstrates the Responsible AI features of the framework,
including bias detection and fairness metrics evaluation.

Author: XAI-SHAP Framework
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.core.framework import XAIFramework
from src.evaluation.fairness import FairnessEvaluator, BiasDetector
from src.data_processing.bias_detector import BiasDetector as DataBiasDetector


def main():
    """
    Complete example of fairness analysis using the XAI-SHAP framework.
    
    This example covers:
    1. Creating synthetic data with potential bias
    2. Detecting bias in data
    3. Training a model
    4. Evaluating fairness metrics
    5. Analyzing SHAP values for fairness insights
    """
    
    print("=" * 60)
    print("XAI-SHAP Framework - Responsible AI / Fairness Analysis")
    print("=" * 60)
    
    # =========================================
    # Step 1: Create Synthetic Data with Bias
    # =========================================
    print("\n📊 Step 1: Creating synthetic data with potential bias...")
    
    np.random.seed(42)
    n_samples = 2000
    
    # Create synthetic loan approval dataset
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'employment_years': np.random.randint(0, 40, n_samples),
        'debt_ratio': np.random.uniform(0.1, 0.9, n_samples),
        'gender': np.random.choice(['male', 'female'], n_samples, p=[0.55, 0.45]),
    }
    
    df = pd.DataFrame(data)
    
    # Create biased target variable (loan approval)
    # Higher approval for males, higher income, higher credit score
    approval_prob = (
        0.3 * (df['credit_score'] > 650).astype(float) +
        0.3 * (df['income'] > 45000).astype(float) +
        0.2 * (df['gender'] == 'male').astype(float) +  # Intentional bias
        0.2 * (df['debt_ratio'] < 0.5).astype(float)
    )
    
    df['approved'] = (approval_prob + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
    
    print(f"   Created dataset with {n_samples} samples")
    print(f"   Approval rate: {df['approved'].mean():.2%}")
    print(f"   Gender distribution: {dict(df['gender'].value_counts())}")
    
    # =========================================
    # Step 2: Detect Bias in Data
    # =========================================
    print("\n🔍 Step 2: Detecting bias in data...")
    
    bias_detector = DataBiasDetector()
    
    # Analyze protected attribute
    df_encoded = df.copy()
    df_encoded['gender_encoded'] = (df['gender'] == 'male').astype(int)
    
    protected_analysis = bias_detector.analyze_protected_attribute(
        data=df_encoded,
        protected_col='gender_encoded',
        target_col='approved'
    )
    
    print("\n   Protected Attribute Analysis (Gender):")
    print(f"   - Value 0 (Female) count: {(df['gender'] == 'female').sum()}")
    print(f"   - Value 1 (Male) count: {(df['gender'] == 'male').sum()}")
    
    # Calculate approval rates by gender
    approval_by_gender = df.groupby('gender')['approved'].mean()
    print(f"\n   Approval rates by gender:")
    for gender, rate in approval_by_gender.items():
        print(f"   - {gender}: {rate:.2%}")
    
    # =========================================
    # Step 3: Prepare Data and Train Model
    # =========================================
    print("\n🤖 Step 3: Training model...")
    
    # Prepare features (encode gender)
    df_model = df.copy()
    df_model['gender_encoded'] = (df_model['gender'] == 'male').astype(int)
    df_model = df_model.drop('gender', axis=1)
    
    # Use framework
    framework = XAIFramework()
    framework.load_data(
        data_path=df_model,
        target_column='approved',
        test_size=0.3
    )
    
    framework.train_model(
        model_type='xgboost',
        model_params={'n_estimators': 100, 'max_depth': 4, 'random_state': 42}
    )
    
    # Get predictions
    y_pred = framework.model.predict(framework.X_test)
    
    print(f"   Model trained successfully!")
    
    # =========================================
    # Step 4: Evaluate Fairness Metrics
    # =========================================
    print("\n⚖️ Step 4: Evaluating fairness metrics...")
    
    # Get protected attribute for test set
    test_indices = df_model.index[-len(framework.X_test):]  # Approximate
    protected_attr = framework.X_test[:, -1]  # gender_encoded is last column
    
    fairness_evaluator = FairnessEvaluator()
    
    # Demographic Parity
    dp = fairness_evaluator.demographic_parity(y_pred, protected_attr)
    print(f"\n   Demographic Parity:")
    print(f"   - Group 0 (Female) positive rate: {dp['group_0_positive_rate']:.4f}")
    print(f"   - Group 1 (Male) positive rate: {dp['group_1_positive_rate']:.4f}")
    print(f"   - Difference: {dp['demographic_parity_difference']:.4f}")
    print(f"   - Ratio: {dp['demographic_parity_ratio']:.4f}")
    
    # Equalized Odds (if we have true labels for test set)
    eo = fairness_evaluator.equalized_odds(framework.y_test, y_pred, protected_attr)
    print(f"\n   Equalized Odds:")
    print(f"   - TPR difference: {eo['true_positive_rate_difference']:.4f}")
    print(f"   - FPR difference: {eo['false_positive_rate_difference']:.4f}")
    
    # Disparate Impact
    di = fairness_evaluator.disparate_impact(y_pred, protected_attr)
    print(f"\n   Disparate Impact:")
    print(f"   - Ratio: {di['disparate_impact_ratio']:.4f}")
    
    if di['disparate_impact_ratio'] < 0.8:
        print("   ⚠️ WARNING: Potential adverse impact (ratio < 0.8)")
    else:
        print("   ✅ No adverse impact detected (ratio >= 0.8)")
    
    # =========================================
    # Step 5: SHAP Analysis for Fairness
    # =========================================
    print("\n🔍 Step 5: SHAP analysis for fairness insights...")
    
    framework.explain_model(n_samples=200)
    
    # Analyze SHAP values for protected attribute
    feature_names = list(df_model.drop('approved', axis=1).columns)
    gender_idx = feature_names.index('gender_encoded')
    
    gender_shap = framework.shap_values[:, gender_idx]
    
    print(f"\n   SHAP values for 'gender_encoded' feature:")
    print(f"   - Mean SHAP value: {np.mean(gender_shap):.4f}")
    print(f"   - Mean absolute SHAP value: {np.mean(np.abs(gender_shap)):.4f}")
    print(f"   - Max SHAP value: {np.max(gender_shap):.4f}")
    print(f"   - Min SHAP value: {np.min(gender_shap):.4f}")
    
    # Feature importance ranking
    mean_abs_shap = np.abs(framework.shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    
    print(f"\n   Feature importance ranking:")
    for i, idx in enumerate(sorted_idx):
        importance = mean_abs_shap[idx]
        name = feature_names[idx]
        protected_marker = " ⚠️" if name == 'gender_encoded' else ""
        print(f"   {i+1}. {name}: {importance:.4f}{protected_marker}")
    
    # =========================================
    # Step 6: Recommendations
    # =========================================
    print("\n" + "=" * 60)
    print("📋 Fairness Assessment Summary")
    print("=" * 60)
    
    issues = []
    recommendations = []
    
    if dp['demographic_parity_difference'] > 0.1:
        issues.append("Significant difference in approval rates between groups")
        recommendations.append("Consider re-sampling or re-weighting training data")
    
    if di['disparate_impact_ratio'] < 0.8:
        issues.append("Disparate impact detected (4/5ths rule violation)")
        recommendations.append("Consider removing or reducing influence of protected attribute")
    
    if mean_abs_shap[gender_idx] > 0.1:
        issues.append("Protected attribute has significant influence on predictions")
        recommendations.append("Evaluate if gender is a valid feature for this use case")
    
    if issues:
        print("\n⚠️ Issues Detected:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        
        print("\n💡 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    else:
        print("\n✅ No significant fairness issues detected!")
    
    print("\n" + "=" * 60)
    print("✨ Fairness analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
