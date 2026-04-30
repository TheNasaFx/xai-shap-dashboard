"""
Analysis Module - Судлаачдад зориулсан гүнзгий шинжилгээний хэрэгслүүд
======================================================================

Энэ модуль нь XAI-SHAP framework-ийн судалгааны түвшний
шинжилгээний боломжуудыг хангана:

- InsightGenerator: Автомат дүгнэлт үүсгэх
- StabilityAnalyzer: SHAP утгын тогтвортой байдлын шинжилгээ
- CounterfactualGenerator: "Юу өөрчлөгдвөл" тайлбар
- ModelComparator: Олон загварын харьцуулалт
- ErrorAnalyzer: Алдааны гүнзгий шинжилгээ

Зохиогч: XAI-SHAP Framework
"""

from src.analysis.insight_generator import InsightGenerator
from src.analysis.stability import StabilityAnalyzer
from src.analysis.counterfactual import CounterfactualGenerator
from src.analysis.model_comparator import ModelComparator
from src.analysis.error_analyzer import ErrorAnalyzer

__all__ = [
    'InsightGenerator',
    'StabilityAnalyzer', 
    'CounterfactualGenerator',
    'ModelComparator',
    'ErrorAnalyzer'
]
