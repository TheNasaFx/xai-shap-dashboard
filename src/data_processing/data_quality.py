"""
Data Quality Analyzer - Өгөгдлийн чанарын автомат шинжилгээ
============================================================

Өгөгдлийг ачаалахаас өмнө болон дараа нь чанарын
иж бүрэн шинжилгээ хийж, асуудлуудыг илрүүлнэ.

Зохиогч: XAI-SHAP Framework
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import warnings
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from scipy import stats

logger = logging.getLogger(__name__)


class DataQualityAnalyzer:
    """
    Өгөгдлийн чанарын иж бүрэн шинжилгээний класс.
    
    Шинжилдэг зүйлс:
    - Missing values (дутуу утгууд)
    - Outliers (гажуудлууд)
    - Class imbalance (анги тэнцвэргүй байдал)
    - Feature types (шинж чанарын төрлүүд)
    - Duplicates (давхардал)
    - Constant features (тогтмол шинж чанарууд)
    - High correlation (өндөр корреляци)
    - Skewness (хазайлт)
    
    Жишээ:
        >>> analyzer = DataQualityAnalyzer()
        >>> report = analyzer.analyze(df, target='label')
        >>> print(report['summary'])
        >>> analyzer.visualize_report(report)
    """
    
    def __init__(self, config=None):
        """
        DataQualityAnalyzer эхлүүлэх.
        
        Параметрүүд:
            config: Тохиргооны менежер instance
        """
        self.config = config
        
        # Тохиргоонууд
        self.outlier_threshold = 3.0  # Z-score threshold
        self.correlation_threshold = 0.95  # High correlation threshold
        self.imbalance_threshold = 0.3  # Class ratio threshold
        self.missing_threshold = 0.5  # 50% missing = сануулга
        self.variance_threshold = 0.01  # Quasi-constant threshold
        
    def analyze(
        self,
        data: pd.DataFrame,
        target: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Өгөгдлийн чанарын иж бүрэн шинжилгээ хийх.
        
        Параметрүүд:
            data: Шинжлэх DataFrame
            target: Зорилтот баганы нэр (заавал биш)
            
        Буцаах:
            Шинжилгээний үр дүнтэй dictionary
        """
        logger.info(f"Өгөгдлийн чанарын шинжилгээ эхэлж байна. Хэмжээ: {data.shape}")
        
        report = {
            'overview': self._analyze_overview(data),
            'missing_values': self._analyze_missing(data),
            'outliers': self._analyze_outliers(data),
            'duplicates': self._analyze_duplicates(data),
            'feature_types': self._analyze_feature_types(data),
            'structural_risks': self._analyze_structural_risks(data),
            'constant_features': self._analyze_constant_features(data),
            'correlation': self._analyze_correlation(data, target),
            'skewness': self._analyze_skewness(data),
            'target_analysis': None,
            'leakage_risks': {
                'risk_count': 0,
                'high_risk_count': 0,
                'details': []
            },
            'recommendations': []
        }
        
        # Class balance шинжилгээ (хэрэв target байвал)
        if target and target in data.columns:
            report['target_analysis'] = self._analyze_target_column(data, target)
            if report['target_analysis']['task_type'] == 'classification':
                report['class_balance'] = self._analyze_class_balance(data, target)
            report['leakage_risks'] = self._analyze_leakage_risks(data, target)
        
        # Зөвлөмжүүд үүсгэх
        report['recommendations'] = self._generate_recommendations(report)
        
        # Нийт оноо
        report['quality_score'] = self._calculate_quality_score(report)
        report['summary'] = self.get_summary_text(report)
        
        logger.info(f"Шинжилгээ дууслаа. Чанарын оноо: {report['quality_score']:.1f}/100")
        
        return report

    def analyze_target_column(
        self,
        data: pd.DataFrame,
        target: str
    ) -> Dict[str, Any]:
        """Target column suitability-г UI preview-д зориулж шалгах."""
        return self._analyze_target_column(data, target)
    
    def _analyze_overview(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Өгөгдлийн ерөнхий тойм."""
        total_rows = len(data)
        total_columns = len(data.columns)
        total_cells = total_rows * total_columns
        size_warning = None
        size_message = None

        if total_rows >= 100_000 or total_cells >= 5_000_000:
            size_warning = 'high'
            size_message = 'Өгөгдлийн хэмжээ маш том байна. SHAP болон multi-model analysis удаашрах магадлалтай.'
        elif total_rows >= 20_000 or total_cells >= 1_000_000:
            size_warning = 'medium'
            size_message = 'Өгөгдлийн хэмжээ дунджаас их байна. Heavy visualization болон SHAP analysis хийхдээ sample strategy бодолцоно уу.'

        return {
            'total_rows': total_rows,
            'total_columns': total_columns,
            'total_cells': total_cells,
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(data.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': len(data.select_dtypes(include=['datetime64']).columns),
            'size_warning': size_warning,
            'size_message': size_message
        }
    
    def _analyze_missing(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Дутуу утгуудын шинжилгээ."""
        missing_counts = data.isnull().sum()
        missing_percent = (missing_counts / len(data)) * 100
        
        missing_details = {}
        for col in data.columns:
            if missing_counts[col] > 0:
                missing_details[col] = {
                    'count': int(missing_counts[col]),
                    'percent': float(missing_percent[col]),
                    'pattern': self._detect_missing_pattern(data[col])
                }
        
        return {
            'total_missing_cells': int(missing_counts.sum()),
            'total_missing_percent': float((missing_counts.sum() / data.size) * 100),
            'columns_with_missing': len(missing_details),
            'details': missing_details
        }
    
    def _detect_missing_pattern(self, series: pd.Series) -> str:
        """Missing pattern илрүүлэх (MCAR, MAR, MNAR)."""
        # Энгийн эвристик - бодит шинжилгээнд илүү нарийн арга хэрэглэнэ
        missing_pct = series.isnull().mean()
        if missing_pct < 0.05:
            return "MCAR (бага хэмжээ)"
        elif missing_pct < 0.2:
            return "MAR (дунд хэмжээ)"
        else:
            return "MNAR (их хэмжээ - шалгах шаардлагатай)"
    
    def _analyze_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Outlier шинжилгээ (Z-score ба IQR аргууд)."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_details = {}
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) < 10:
                continue
            
            # Z-score арга
            z_scores = np.abs(stats.zscore(col_data))
            z_outliers = int((z_scores > self.outlier_threshold).sum())
            
            # IQR арга
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = int(((col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)).sum())
            
            if z_outliers > 0 or iqr_outliers > 0:
                outlier_details[col] = {
                    'z_score_outliers': z_outliers,
                    'iqr_outliers': iqr_outliers,
                    'percent': float(max(z_outliers, iqr_outliers) / len(col_data) * 100),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std())
                }
        
        return {
            'total_columns_with_outliers': len(outlier_details),
            'details': outlier_details
        }
    
    def _analyze_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Давхардсан мөрүүдийн шинжилгээ."""
        duplicates = data.duplicated()
        duplicate_count = duplicates.sum()
        
        return {
            'duplicate_rows': int(duplicate_count),
            'duplicate_percent': float(duplicate_count / len(data) * 100),
            'unique_rows': int(len(data) - duplicate_count)
        }
    
    def _analyze_feature_types(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Шинж чанаруудын төрлийн шинжилгээ."""
        type_info = {}
        
        for col in data.columns:
            dtype = str(data[col].dtype)
            unique_count = data[col].nunique()
            
            # Төрөл тодорхойлох
            if data[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                if unique_count <= 10:
                    feature_type = 'numeric_discrete'
                else:
                    feature_type = 'numeric_continuous'
            elif data[col].dtype == 'object':
                if unique_count <= 2:
                    feature_type = 'binary'
                elif unique_count <= 20:
                    feature_type = 'categorical_low_cardinality'
                else:
                    feature_type = 'categorical_high_cardinality'
            elif data[col].dtype == 'bool':
                feature_type = 'binary'
            else:
                feature_type = 'other'
            
            type_info[col] = {
                'dtype': dtype,
                'feature_type': feature_type,
                'unique_count': int(unique_count),
                'sample_values': data[col].dropna().head(3).tolist()
            }
        
        return type_info

    def _analyze_structural_risks(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Date-like, mixed-type, numeric-text, identifier-like columns илрүүлэх."""
        date_like_columns = []
        numeric_text_columns = []
        mixed_type_columns = []
        identifier_like_columns = []

        for col in data.columns:
            series = data[col].dropna()
            if len(series) < 5:
                continue

            sample = series if len(series) <= 2000 else series.sample(2000, random_state=42)
            sample_as_text = sample.astype(str).str.strip()
            sample_as_text = sample_as_text[sample_as_text != '']
            if sample_as_text.empty:
                continue

            unique_ratio = float(series.nunique(dropna=True) / max(len(series), 1))
            col_name = str(col).lower()
            if unique_ratio >= 0.95 and any(token in col_name for token in ('id', 'uuid', 'key', 'code', 'number', 'no')):
                identifier_like_columns.append({
                    'column': col,
                    'unique_ratio': unique_ratio,
                    'message': 'Identifier-like багана байж магадгүй. Model feature болговол leakage эсвэл overfitting үүсгэж болно.'
                })

            if is_numeric_dtype(series) or is_bool_dtype(series) or is_datetime64_any_dtype(series):
                continue

            numeric_ratio = float(pd.to_numeric(sample_as_text, errors='coerce').notna().mean())
            if numeric_ratio >= 0.85:
                numeric_text_columns.append({
                    'column': col,
                    'numeric_ratio': numeric_ratio,
                    'message': 'Numeric value-ууд text хэлбэрээр орж ирсэн байна.'
                })
            elif 0.15 <= numeric_ratio < 0.85:
                mixed_type_columns.append({
                    'column': col,
                    'numeric_ratio': numeric_ratio,
                    'message': 'Text болон numeric утгууд холилдсон байж магадгүй.'
                })

            date_pattern_ratio = float(sample_as_text.str.contains(r'[-/:]', regex=True).mean())
            if date_pattern_ratio >= 0.6:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    parsed_ratio = float(pd.to_datetime(sample_as_text, errors='coerce').notna().mean())
                if parsed_ratio >= 0.8:
                    date_like_columns.append({
                        'column': col,
                        'parsed_ratio': parsed_ratio,
                        'message': 'Date/time information text хэлбэрээр байна. Temporal feature engineering хэрэгтэй байж болно.'
                    })

        return {
            'date_like_columns': date_like_columns,
            'date_like_count': len(date_like_columns),
            'numeric_text_columns': numeric_text_columns,
            'numeric_text_count': len(numeric_text_columns),
            'mixed_type_columns': mixed_type_columns,
            'mixed_type_count': len(mixed_type_columns),
            'identifier_like_columns': identifier_like_columns,
            'identifier_like_count': len(identifier_like_columns)
        }
    
    def _analyze_constant_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Тогтмол болон бараг тогтмол шинж чанаруудын шинжилгээ."""
        constant_features = []
        quasi_constant_features = []
        
        for col in data.columns:
            unique_ratio = data[col].nunique() / len(data)
            
            if data[col].nunique() == 1:
                constant_features.append(col)
            elif unique_ratio < self.variance_threshold:
                dominant_pct = data[col].value_counts(normalize=True).iloc[0] * 100
                quasi_constant_features.append({
                    'column': col,
                    'dominant_value': data[col].mode().iloc[0] if len(data[col].mode()) > 0 else None,
                    'dominant_percent': float(dominant_pct)
                })
        
        return {
            'constant_features': constant_features,
            'constant_count': len(constant_features),
            'quasi_constant_features': quasi_constant_features,
            'quasi_constant_count': len(quasi_constant_features)
        }
    
    def _analyze_correlation(
        self, 
        data: pd.DataFrame,
        target: Optional[str] = None
    ) -> Dict[str, Any]:
        """Корреляцийн шинжилгээ."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return {'high_correlations': [], 'correlation_matrix': None}
        
        corr_matrix = numeric_data.corr()
        
        # Өндөр корреляцтай хосуудыг олох
        high_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= self.correlation_threshold:
                    high_correlations.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': float(corr_val)
                    })
        
        # Target-тэй корреляци
        target_correlations = {}
        if target and target in numeric_data.columns:
            for col in numeric_data.columns:
                if col != target:
                    target_correlations[col] = float(corr_matrix.loc[col, target])
        
        return {
            'high_correlations': high_correlations,
            'high_correlation_count': len(high_correlations),
            'target_correlations': target_correlations,
            'correlation_matrix': corr_matrix.round(3).to_dict() if len(corr_matrix) <= 20 else None
        }
    
    def _analyze_skewness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Хазайлтын шинжилгээ."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        skewness_details = {}
        
        for col in numeric_cols:
            skew = float(data[col].skew())
            if abs(skew) > 1:
                skewness_details[col] = {
                    'skewness': skew,
                    'severity': 'high' if abs(skew) > 2 else 'moderate',
                    'direction': 'right' if skew > 0 else 'left',
                    'recommendation': 'log_transform' if skew > 1 else 'sqrt_transform'
                }
        
        return {
            'skewed_features': list(skewness_details.keys()),
            'skewed_count': len(skewness_details),
            'details': skewness_details
        }
    
    def _analyze_class_balance(
        self, 
        data: pd.DataFrame, 
        target: str
    ) -> Dict[str, Any]:
        """Анги тэнцвэрийн шинжилгээ."""
        class_counts = data[target].value_counts()
        class_ratios = data[target].value_counts(normalize=True)
        
        min_class = class_counts.idxmin()
        max_class = class_counts.idxmax()
        imbalance_ratio = class_counts[min_class] / class_counts[max_class]
        
        is_imbalanced = imbalance_ratio < self.imbalance_threshold
        
        return {
            'class_distribution': class_counts.to_dict(),
            'class_percentages': {k: float(v * 100) for k, v in class_ratios.items()},
            'minority_class': str(min_class),
            'majority_class': str(max_class),
            'imbalance_ratio': float(imbalance_ratio),
            'is_imbalanced': is_imbalanced,
            'recommendation': 'SMOTE эсвэл class weights ашиглах' if is_imbalanced else None
        }

    def _analyze_target_column(
        self,
        data: pd.DataFrame,
        target: str
    ) -> Dict[str, Any]:
        """Target column судалгаанд тохиромжтой эсэхийг шалгах."""
        target_series = data[target]
        non_null = target_series.dropna()
        missing_percent = float(target_series.isnull().mean() * 100)
        unique_count = int(non_null.nunique(dropna=True))
        unique_ratio = float(unique_count / max(len(non_null), 1))
        issues = []

        task_type = 'unsupported'
        recommended_metric = 'Target redesign шаардлагатай'
        integer_like = False

        if is_numeric_dtype(non_null):
            numeric_non_null = pd.to_numeric(non_null, errors='coerce').dropna()
            integer_like = bool(np.allclose(numeric_non_null % 1, 0)) if not numeric_non_null.empty else False

        if missing_percent > 0:
            issues.append({
                'severity': 'medium' if missing_percent <= 5 else 'high',
                'message': f'Target баганад {missing_percent:.1f}% missing value байна.'
            })

        if unique_count <= 1:
            issues.append({'severity': 'high', 'message': 'Target багана тогтмол байна. Загвар сургах боломжгүй.'})
        elif unique_ratio >= 0.95:
            issues.append({'severity': 'high', 'message': 'Target бараг row бүрт unique байна. ID маягийн target эсвэл label issue байж магадгүй.'})

        if unique_count > 1 and unique_ratio < 0.95:
            if is_bool_dtype(non_null):
                task_type = 'classification'
                recommended_metric = 'Accuracy / F1 / ROC-AUC'
            elif is_numeric_dtype(non_null):
                if integer_like and unique_count <= min(20, max(10, int(len(non_null) * 0.1))):
                    task_type = 'classification'
                    recommended_metric = 'Weighted F1 / Balanced Accuracy'
                else:
                    task_type = 'regression'
                    recommended_metric = 'RMSE / MAE / R²'
            elif unique_count <= min(50, max(10, int(len(non_null) * 0.2))):
                task_type = 'classification'
                recommended_metric = 'Weighted F1'
            else:
                issues.append({'severity': 'high', 'message': 'Target классын тоо хэт олон байна. High-cardinality label байж магадгүй.'})

        if task_type == 'classification' and unique_count > 20:
            issues.append({'severity': 'medium', 'message': f'Classification target {unique_count} class-тай байна. Metric болон class balance interpretation илүү төвөгтэй болно.'})

        if task_type == 'regression' and integer_like and unique_count <= 5:
            issues.append({'severity': 'medium', 'message': 'Numeric target хэдхэн discrete утгатай байна. Энэ нь regression биш classification байж магадгүй.'})

        is_suitable = task_type != 'unsupported' and not any(issue['severity'] == 'high' for issue in issues)

        return {
            'target': target,
            'dtype': str(target_series.dtype),
            'missing_percent': missing_percent,
            'unique_count': unique_count,
            'unique_ratio': unique_ratio,
            'task_type': task_type,
            'recommended_metric': recommended_metric,
            'is_suitable': is_suitable,
            'issues': issues
        }

    def _analyze_leakage_risks(
        self,
        data: pd.DataFrame,
        target: str
    ) -> Dict[str, Any]:
        """Target leakage heuristic шалгалтууд."""
        target_series = data[target]
        risks = []
        target_tokens = {tok for tok in re.split(r'[_\W]+', str(target).lower()) if len(tok) >= 3}

        for col in data.columns:
            if col == target:
                continue

            feature = data[col]
            aligned = pd.DataFrame({'feature': feature, 'target': target_series}).dropna()
            if len(aligned) < 5:
                continue

            feature_text = aligned['feature'].astype(str).str.strip().str.lower()
            target_text = aligned['target'].astype(str).str.strip().str.lower()
            exact_match_ratio = float((feature_text == target_text).mean())
            if exact_match_ratio >= 0.98:
                risks.append({
                    'feature': col,
                    'risk_type': 'exact_match',
                    'severity': 'high',
                    'evidence': f'{exact_match_ratio:.1%} утга target-тай яг ижил байна',
                    'message': 'Feature target-ийг шууд хуулсан эсвэл label leakage байж магадгүй.'
                })
                continue

            if is_numeric_dtype(feature) and is_numeric_dtype(target_series):
                numeric_pair = pd.concat([feature, target_series], axis=1).dropna()
                if len(numeric_pair) >= 5 and numeric_pair.iloc[:, 0].nunique() > 1 and numeric_pair.iloc[:, 1].nunique() > 1:
                    corr = float(abs(numeric_pair.corr().iloc[0, 1]))
                    if corr >= 0.98:
                        risks.append({
                            'feature': col,
                            'risk_type': 'near_perfect_correlation',
                            'severity': 'high',
                            'evidence': f'|corr(target)| = {corr:.3f}',
                            'message': 'Target-тай бараг төгс хамааралтай feature илэрлээ. Leakage эсвэл target-derived feature байж болно.'
                        })

            feature_tokens = {tok for tok in re.split(r'[_\W]+', str(col).lower()) if len(tok) >= 3}
            shared_tokens = sorted(target_tokens & feature_tokens)
            if shared_tokens:
                risks.append({
                    'feature': col,
                    'risk_type': 'name_overlap',
                    'severity': 'medium',
                    'evidence': f'Нэр давхцсан token: {", ".join(shared_tokens[:3])}',
                    'message': 'Feature нэр target-тай хүчтэй төстэй байна. Target-derived feature эсэхийг шалгана уу.'
                })

        severity_rank = {'high': 0, 'medium': 1, 'low': 2}
        risks.sort(key=lambda item: (severity_rank.get(item['severity'], 3), item['feature'], item['risk_type']))

        return {
            'risk_count': len(risks),
            'high_risk_count': sum(1 for item in risks if item['severity'] == 'high'),
            'details': risks
        }
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[Dict[str, str]]:
        """Шинжилгээнээс зөвлөмжүүд үүсгэх."""
        recommendations = []
        
        # Missing values
        missing = report['missing_values']
        if missing['columns_with_missing'] > 0:
            high_missing = [col for col, info in missing['details'].items() 
                          if info['percent'] > 20]
            if high_missing:
                recommendations.append({
                    'category': 'missing_values',
                    'priority': 'high',
                    'message': f"{len(high_missing)} багана 20%-аас их дутуу утгатай: {', '.join(high_missing[:3])}",
                    'action': 'Эдгээр баганыг хасах эсвэл imputation хийх'
                })
        
        # Outliers
        outliers = report['outliers']
        if outliers['total_columns_with_outliers'] > 0:
            severe_outliers = [col for col, info in outliers['details'].items()
                            if info['percent'] > 5]
            if severe_outliers:
                recommendations.append({
                    'category': 'outliers',
                    'priority': 'medium',
                    'message': f"{len(severe_outliers)} багана их хэмжээний outlier-тэй",
                    'action': 'RobustScaler ашиглах эсвэл outlier-уудыг зохицуулах'
                })
        
        # Duplicates
        if report['duplicates']['duplicate_percent'] > 1:
            recommendations.append({
                'category': 'duplicates',
                'priority': 'high',
                'message': f"{report['duplicates']['duplicate_rows']} давхардсан мөр олдсон ({report['duplicates']['duplicate_percent']:.1f}%)",
                'action': 'Давхардсан мөрүүдийг хасах'
            })
        
        # High correlation
        if report['correlation']['high_correlation_count'] > 0:
            recommendations.append({
                'category': 'correlation',
                'priority': 'medium',
                'message': f"{report['correlation']['high_correlation_count']} хос шинж чанар өндөр корреляцтай (>{self.correlation_threshold})",
                'action': 'Нэгийг нь хасах эсвэл PCA ашиглах'
            })
        
        # Constant features
        const = report['constant_features']
        if const['constant_count'] > 0:
            recommendations.append({
                'category': 'constant_features',
                'priority': 'high',
                'message': f"{const['constant_count']} тогтмол шинж чанар олдсон: {', '.join(const['constant_features'][:3])}",
                'action': 'Эдгээр шинж чанаруудыг хасах'
            })
        
        # Class imbalance
        if 'class_balance' in report and report['class_balance']['is_imbalanced']:
            recommendations.append({
                'category': 'class_balance',
                'priority': 'high',
                'message': f"Өгөгдөл тэнцвэргүй (ratio: {report['class_balance']['imbalance_ratio']:.2f})",
                'action': report['class_balance']['recommendation']
            })

        # Structural risks
        structural = report['structural_risks']
        if structural['numeric_text_count'] > 0:
            recommendations.append({
                'category': 'numeric_text',
                'priority': 'high',
                'message': f"{structural['numeric_text_count']} багана numeric утгаа text хэлбэрээр хадгалсан байна",
                'action': 'Эдгээр баганыг numeric dtype рүү хөрвүүлж preprocessing-ийг стандартчилна уу'
            })
        if structural['mixed_type_count'] > 0:
            recommendations.append({
                'category': 'mixed_type',
                'priority': 'high',
                'message': f"{structural['mixed_type_count']} баганад mixed text/numeric утга илэрлээ",
                'action': 'Value cleaning хийж нэг төрлийн schema болгоно уу'
            })
        if structural['date_like_count'] > 0:
            recommendations.append({
                'category': 'date_like',
                'priority': 'medium',
                'message': f"{structural['date_like_count']} багана date-like text байна",
                'action': 'Datetime feature engineering ашиглах эсэхээ шалгана уу'
            })

        # Target suitability
        target_analysis = report.get('target_analysis')
        if target_analysis:
            for issue in target_analysis.get('issues', []):
                recommendations.append({
                    'category': 'target_analysis',
                    'priority': issue['severity'],
                    'message': issue['message'],
                    'action': 'Target column-оо дахин шалгаж, modeling objective-оо тодорхой болгоно уу'
                })

        # Leakage risk
        leakage = report.get('leakage_risks', {})
        if leakage.get('risk_count', 0) > 0:
            recommendations.append({
                'category': 'leakage',
                'priority': 'high' if leakage.get('high_risk_count', 0) > 0 else 'medium',
                'message': f"{leakage.get('risk_count', 0)} leakage heuristic signal илэрлээ",
                'action': 'Эдгээр feature-үүд target-тэй шууд холбоотой эсэхийг training-ээс өмнө нягтална уу'
            })

        # Dataset size
        overview = report['overview']
        if overview.get('size_warning'):
            recommendations.append({
                'category': 'dataset_size',
                'priority': 'medium' if overview['size_warning'] == 'high' else 'low',
                'message': overview['size_message'],
                'action': 'Sampling, lightweight plots, эсвэл staged analysis ашиглахыг бодолцоно уу'
            })
        
        # Skewness
        if report['skewness']['skewed_count'] > 3:
            recommendations.append({
                'category': 'skewness',
                'priority': 'low',
                'message': f"{report['skewness']['skewed_count']} шинж чанар их хазайлттай",
                'action': 'Log эсвэл sqrt transform хийх'
            })
        
        return sorted(recommendations, key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x['priority']])
    
    def _calculate_quality_score(self, report: Dict[str, Any]) -> float:
        """Өгөгдлийн чанарын нийт оноо тооцох (0-100)."""
        score = 100.0
        
        # Missing values penalty (-20 max)
        missing_pct = report['missing_values']['total_missing_percent']
        score -= min(20, missing_pct * 2)
        
        # Duplicates penalty (-10 max)
        dup_pct = report['duplicates']['duplicate_percent']
        score -= min(10, dup_pct)
        
        # High correlation penalty (-10 max)
        high_corr = report['correlation']['high_correlation_count']
        score -= min(10, high_corr * 2)
        
        # Constant features penalty (-10 max)
        const_count = report['constant_features']['constant_count']
        score -= min(10, const_count * 5)
        
        # Outliers penalty (-15 max)
        outlier_cols = report['outliers']['total_columns_with_outliers']
        score -= min(15, outlier_cols * 1.5)
        
        # Class imbalance penalty (-15 max)
        if 'class_balance' in report and report['class_balance']['is_imbalanced']:
            imb_ratio = report['class_balance']['imbalance_ratio']
            score -= min(15, (1 - imb_ratio) * 20)

        # Leakage risk penalty (-25 max)
        leakage = report.get('leakage_risks', {})
        score -= min(
            25,
            leakage.get('high_risk_count', 0) * 12
            + max(leakage.get('risk_count', 0) - leakage.get('high_risk_count', 0), 0) * 4
        )

        # Structural risk penalty (-10 max)
        structural = report.get('structural_risks', {})
        score -= min(10, structural.get('numeric_text_count', 0) * 2 + structural.get('mixed_type_count', 0) * 3)

        # Target suitability penalty (-20 max)
        target_analysis = report.get('target_analysis')
        if target_analysis:
            high_issues = sum(1 for issue in target_analysis.get('issues', []) if issue['severity'] == 'high')
            medium_issues = sum(1 for issue in target_analysis.get('issues', []) if issue['severity'] == 'medium')
            score -= min(20, high_issues * 10 + medium_issues * 4)
        
        return max(0, score)
    
    def get_summary_text(self, report: Dict[str, Any]) -> str:
        """Хэрэглэгчид ээлтэй хураангуй текст үүсгэх."""
        lines = [
            "═" * 60,
            "           ӨГӨГДЛИЙН ЧАНАРЫН ШИНЖИЛГЭЭ",
            "═" * 60,
            "",
            f"[DATA] Ерөнхий мэдээлэл:",
            f"   • Нийт мөр: {report['overview']['total_rows']:,}",
            f"   • Нийт багана: {report['overview']['total_columns']}",
            f"   • Санах ойн хэмжээ: {report['overview']['memory_usage_mb']:.2f} MB",
            "",
            f"[SCORE] Чанарын оноо: {report['quality_score']:.1f}/100",
            ""
        ]

        overview = report['overview']
        if overview.get('size_message'):
            lines.append(f"[!] Dataset хэмжээ: {overview['size_message']}")
        
        # Missing values
        missing = report['missing_values']
        if missing['columns_with_missing'] > 0:
            lines.append(f"[!] Дутуу утгууд: {missing['columns_with_missing']} багана ({missing['total_missing_percent']:.1f}%)")
        else:
            lines.append(f"[OK] Дутуу утга: Байхгүй")
        
        # Duplicates
        dups = report['duplicates']
        if dups['duplicate_rows'] > 0:
            lines.append(f"[!] Давхардсан мөр: {dups['duplicate_rows']} ({dups['duplicate_percent']:.1f}%)")
        else:
            lines.append(f"[OK] Давхардал: Байхгүй")
        
        # Outliers
        outliers = report['outliers']
        if outliers['total_columns_with_outliers'] > 0:
            lines.append(f"[!] Outlier-тэй баганууд: {outliers['total_columns_with_outliers']}")
        else:
            lines.append(f"[OK] Outlier: Хэвийн")
        
        # Class balance
        if 'class_balance' in report:
            cb = report['class_balance']
            if cb['is_imbalanced']:
                lines.append(f"[!] Анги тэнцвэргүй: ratio = {cb['imbalance_ratio']:.2f}")
            else:
                lines.append(f"[OK] Анги тэнцвэр: Хэвийн")

        target_analysis = report.get('target_analysis')
        if target_analysis:
            if target_analysis['is_suitable']:
                lines.append(f"[OK] Target тохиромжтой: {target_analysis['task_type']}")
            else:
                lines.append(f"[!] Target эрсдэлтэй: {target_analysis['task_type']}")

        leakage = report.get('leakage_risks', {})
        if leakage.get('risk_count', 0) > 0:
            lines.append(f"[!] Leakage signal: {leakage['risk_count']} feature")

        structural = report.get('structural_risks', {})
        if structural.get('numeric_text_count', 0) > 0 or structural.get('mixed_type_count', 0) > 0:
            lines.append(
                f"[!] Structural risk: numeric-text={structural.get('numeric_text_count', 0)}, mixed-type={structural.get('mixed_type_count', 0)}"
            )
        
        # Recommendations
        recs = report['recommendations']
        if recs:
            lines.extend(["", "[RECOMMENDATIONS] Зөвлөмжүүд:"])
            for i, rec in enumerate(recs[:5], 1):
                priority_icon = {"high": "[HIGH]", "medium": "[MED]", "low": "[LOW]"}[rec['priority']]
                lines.append(f"   {i}. {priority_icon} {rec['message']}")
        
        lines.append("")
        lines.append("═" * 60)
        
        return "\n".join(lines)
