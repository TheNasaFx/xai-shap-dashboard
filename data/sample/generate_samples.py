"""
Sample Dataset Generator
=========================

Generates sample datasets for testing and demonstration.

Author: XAI-SHAP Framework
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris


def generate_loan_dataset(n_samples: int = 1000, save_path: str = None) -> pd.DataFrame:
    """
    Generate synthetic loan approval dataset.
    
    Args:
        n_samples: Number of samples to generate
        save_path: Optional path to save CSV
        
    Returns:
        DataFrame with loan data
    """
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.normal(55000, 20000, n_samples).clip(15000, 200000),
        'credit_score': np.random.randint(300, 850, n_samples),
        'employment_years': np.random.randint(0, 40, n_samples),
        'loan_amount': np.random.normal(50000, 30000, n_samples).clip(5000, 200000),
        'debt_to_income': np.random.uniform(0.05, 0.8, n_samples),
        'num_credit_lines': np.random.randint(1, 15, n_samples),
        'late_payments': np.random.poisson(1, n_samples),
        'home_ownership': np.random.choice(['rent', 'own', 'mortgage'], n_samples),
        'purpose': np.random.choice(['debt_consolidation', 'home_improvement', 'education', 'business'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate target (approved) based on features
    approval_prob = (
        0.25 * (df['credit_score'] > 650).astype(float) +
        0.2 * (df['income'] > 50000).astype(float) +
        0.2 * (df['debt_to_income'] < 0.4).astype(float) +
        0.15 * (df['employment_years'] > 3).astype(float) +
        0.1 * (df['late_payments'] == 0).astype(float) +
        0.1 * (df['home_ownership'] == 'own').astype(float)
    )
    
    df['approved'] = (approval_prob + np.random.normal(0, 0.15, n_samples) > 0.5).astype(int)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Dataset saved to: {save_path}")
    
    return df


def generate_credit_risk_dataset(n_samples: int = 1000, save_path: str = None) -> pd.DataFrame:
    """
    Generate synthetic credit risk dataset.
    
    Args:
        n_samples: Number of samples
        save_path: Optional save path
        
    Returns:
        DataFrame with credit risk data
    """
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(20, 65, n_samples),
        'annual_income': np.random.lognormal(10.5, 0.5, n_samples),
        'monthly_expenses': np.random.lognormal(8, 0.5, n_samples),
        'credit_history_length': np.random.randint(0, 30, n_samples),
        'num_open_accounts': np.random.randint(1, 20, n_samples),
        'num_delinquencies': np.random.poisson(0.5, n_samples),
        'total_debt': np.random.lognormal(9, 1, n_samples),
        'credit_utilization': np.random.uniform(0, 1, n_samples),
        'employment_status': np.random.choice(['employed', 'self-employed', 'unemployed', 'retired'], n_samples, p=[0.6, 0.2, 0.1, 0.1]),
        'education': np.random.choice(['high_school', 'bachelor', 'master', 'phd'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
    }
    
    df = pd.DataFrame(data)
    
    # Risk score (0 = low risk, 1 = high risk)
    risk_factors = (
        0.3 * (df['num_delinquencies'] > 0).astype(float) +
        0.25 * (df['credit_utilization'] > 0.5).astype(float) +
        0.2 * (df['credit_history_length'] < 5).astype(float) +
        0.15 * (df['employment_status'] == 'unemployed').astype(float) +
        0.1 * (df['total_debt'] / df['annual_income'] > 0.5).astype(float)
    )
    
    df['high_risk'] = (risk_factors + np.random.normal(0, 0.1, n_samples) > 0.4).astype(int)
    
    if save_path:
        df.to_csv(save_path, index=False)
    
    return df


def get_breast_cancer_dataset(save_path: str = None) -> pd.DataFrame:
    """
    Get breast cancer dataset from sklearn.
    
    Args:
        save_path: Optional save path
        
    Returns:
        DataFrame with breast cancer data
    """
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['diagnosis'] = data.target  # 0 = malignant, 1 = benign
    
    if save_path:
        df.to_csv(save_path, index=False)
    
    return df


def generate_employee_attrition_dataset(n_samples: int = 1000, save_path: str = None) -> pd.DataFrame:
    """
    Generate synthetic employee attrition dataset.
    
    Args:
        n_samples: Number of samples
        save_path: Optional save path
        
    Returns:
        DataFrame with employee data
    """
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(22, 60, n_samples),
        'years_at_company': np.random.randint(0, 35, n_samples),
        'monthly_income': np.random.normal(6000, 2000, n_samples).clip(2000, 20000),
        'job_satisfaction': np.random.randint(1, 5, n_samples),
        'work_life_balance': np.random.randint(1, 5, n_samples),
        'overtime': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'distance_from_home': np.random.randint(1, 50, n_samples),
        'num_companies_worked': np.random.randint(0, 10, n_samples),
        'training_times_last_year': np.random.randint(0, 6, n_samples),
        'department': np.random.choice(['Sales', 'R&D', 'HR'], n_samples),
        'education_level': np.random.randint(1, 6, n_samples),
        'performance_rating': np.random.randint(1, 5, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Attrition probability
    attrition_prob = (
        0.3 * (df['job_satisfaction'] < 2).astype(float) +
        0.2 * (df['overtime'] == 1).astype(float) +
        0.2 * (df['work_life_balance'] < 2).astype(float) +
        0.15 * (df['years_at_company'] < 2).astype(float) +
        0.15 * (df['distance_from_home'] > 20).astype(float)
    )
    
    df['attrition'] = (attrition_prob + np.random.normal(0, 0.15, n_samples) > 0.35).astype(int)
    
    if save_path:
        df.to_csv(save_path, index=False)
    
    return df


if __name__ == "__main__":
    import os
    
    # Create sample directory
    os.makedirs("data/sample", exist_ok=True)
    
    # Generate datasets
    print("Generating sample datasets...")
    
    generate_loan_dataset(1000, "data/sample/loan_data.csv")
    generate_credit_risk_dataset(1000, "data/sample/credit_risk.csv")
    get_breast_cancer_dataset("data/sample/breast_cancer.csv")
    generate_employee_attrition_dataset(1000, "data/sample/employee_attrition.csv")
    
    print("\n✅ All sample datasets generated in data/sample/")
