"""
XAI Dashboard-д зориулсан Dataset татах скрипт
===============================================
sklearn болон openml-ээс алдартай dataset-уудыг татаж CSV хадгална.
"""
import pandas as pd
from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings('ignore')

print("Dataset татаж байна...")

# 1. Heart Disease Dataset (303 rows, 13 features)
print("\n[1/3] Heart Disease Dataset татаж байна...")
try:
    heart = fetch_openml(name='heart-statlog', version=1, as_frame=True, parser='auto')
    df_heart = heart.frame
    # Target баганыг binary болгох (class: absent=0, present=1)
    df_heart['target'] = (df_heart['class'] == 'present').astype(int)
    df_heart = df_heart.drop('class', axis=1)
    df_heart.to_csv("heart_disease.csv", index=False)
    print(f"   ✓ heart_disease.csv хадгалагдлаа ({df_heart.shape[0]} мөр, {df_heart.shape[1]} багана)")
except Exception as e:
    print(f"   ✗ Heart Disease татахад алдаа: {e}")

# 2. Bank Marketing Dataset (45,211 rows, 17 features)
print("\n[2/3] Bank Marketing Dataset татаж байна...")
try:
    bank = fetch_openml(name='bank-marketing', version=1, as_frame=True, parser='auto')
    df_bank = bank.frame
    # Target баганыг binary болгох (Class: 1=no, 2=yes -> 0, 1)
    df_bank['y'] = (df_bank['Class'] == '2').astype(int)
    df_bank = df_bank.drop('Class', axis=1)
    df_bank.to_csv("bank_marketing.csv", index=False)
    print(f"   ✓ bank_marketing.csv хадгалагдлаа ({df_bank.shape[0]} мөр, {df_bank.shape[1]} багана)")
except Exception as e:
    print(f"   ✗ Bank Marketing татахад алдаа: {e}")

# 3. Adult Census Dataset (48,842 rows, 14 features)
print("\n[3/3] Adult Census Dataset татаж байна...")
try:
    adult = fetch_openml(name='adult', version=2, as_frame=True, parser='auto')
    df_adult = adult.frame
    # Target баганыг binary болгох (class: <=50K=0, >50K=1)
    df_adult['income'] = (df_adult['class'] == '>50K').astype(int)
    df_adult = df_adult.drop('class', axis=1)
    df_adult.to_csv("adult_census.csv", index=False)
    print(f"   ✓ adult_census.csv хадгалагдлаа ({df_adult.shape[0]} мөр, {df_adult.shape[1]} багана)")
except Exception as e:
    print(f"   ✗ Adult Census татахад алдаа: {e}")

print("\n" + "="*50)
print("Дууслаа! CSV файлуудыг dashboard-т ачаалж болно.")
print("="*50)