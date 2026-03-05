"""
Data Transformers - Specialized Data Transformation Classes
============================================================

Individual transformer classes for specific data transformations.
These can be used independently or composed in pipelines.

Author: XAI-SHAP Framework
"""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Handle missing values in data.
    
    Strategies:
    - 'mean': Fill with column mean
    - 'median': Fill with column median
    - 'mode': Fill with most frequent value
    - 'constant': Fill with specified constant
    - 'drop': Drop rows with missing values
    
    Example:
        >>> handler = MissingValueHandler(strategy='median')
        >>> X_filled = handler.fit_transform(X)
    """
    
    def __init__(
        self,
        strategy: str = 'median',
        fill_value: Optional[Any] = None,
        columns: Optional[List[str]] = None
    ):
        """
        Initialize MissingValueHandler.
        
        Args:
            strategy: Imputation strategy
            fill_value: Value for 'constant' strategy
            columns: Specific columns to process (None = all)
        """
        self.strategy = strategy
        self.fill_value = fill_value
        self.columns = columns
        self._fill_values = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit handler by computing fill values."""
        X = pd.DataFrame(X)
        columns = self.columns or X.columns.tolist()
        
        for col in columns:
            if col not in X.columns:
                continue
                
            if self.strategy == 'mean':
                self._fill_values[col] = X[col].mean()
            elif self.strategy == 'median':
                self._fill_values[col] = X[col].median()
            elif self.strategy == 'mode':
                mode_vals = X[col].mode()
                self._fill_values[col] = mode_vals[0] if len(mode_vals) > 0 else None
            elif self.strategy == 'constant':
                self._fill_values[col] = self.fill_value
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by filling missing values."""
        X = pd.DataFrame(X).copy()
        
        for col, fill_val in self._fill_values.items():
            if col in X.columns and fill_val is not None:
                X[col] = X[col].fillna(fill_val)
        
        if self.strategy == 'drop':
            X = X.dropna()
        
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical variables.
    
    Methods:
    - 'onehot': One-hot encoding
    - 'label': Label encoding
    - 'ordinal': Ordinal encoding with specified order
    - 'target': Target encoding (requires y)
    
    Example:
        >>> encoder = CategoricalEncoder(method='onehot')
        >>> X_encoded = encoder.fit_transform(X)
    """
    
    def __init__(
        self,
        method: str = 'onehot',
        columns: Optional[List[str]] = None,
        drop_first: bool = True,
        ordinal_mappings: Optional[Dict[str, List]] = None
    ):
        """
        Initialize CategoricalEncoder.
        
        Args:
            method: Encoding method
            columns: Columns to encode (None = auto-detect categorical)
            drop_first: Drop first category in one-hot encoding
            ordinal_mappings: Category order for ordinal encoding
        """
        self.method = method
        self.columns = columns
        self.drop_first = drop_first
        self.ordinal_mappings = ordinal_mappings or {}
        self._encoders = {}
        self._encoded_columns = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit encoder to data."""
        X = pd.DataFrame(X)
        
        # Auto-detect categorical columns if not specified
        if self.columns is None:
            self.columns = X.select_dtypes(
                include=['object', 'category']
            ).columns.tolist()
        
        for col in self.columns:
            if col not in X.columns:
                continue
            
            if self.method == 'label':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self._encoders[col] = le
                
            elif self.method == 'onehot':
                categories = X[col].unique().tolist()
                self._encoders[col] = categories
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted encodings."""
        X = pd.DataFrame(X).copy()
        self._encoded_columns = []
        
        for col in self.columns:
            if col not in X.columns or col not in self._encoders:
                continue
            
            if self.method == 'label':
                X[col] = self._encoders[col].transform(X[col].astype(str))
                self._encoded_columns.append(col)
                
            elif self.method == 'onehot':
                dummies = pd.get_dummies(
                    X[col], 
                    prefix=col, 
                    drop_first=self.drop_first
                )
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                self._encoded_columns.extend(dummies.columns.tolist())
        
        return X
    
    def get_feature_names(self) -> List[str]:
        """Get names of encoded features."""
        return self._encoded_columns


class FeatureNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalize numerical features.
    
    Methods:
    - 'standard': StandardScaler (z-score normalization)
    - 'minmax': MinMaxScaler (0-1 scaling)
    - 'robust': RobustScaler (median/IQR based)
    - 'log': Log transformation
    - 'none': No normalization
    
    Example:
        >>> normalizer = FeatureNormalizer(method='standard')
        >>> X_normalized = normalizer.fit_transform(X)
    """
    
    def __init__(
        self,
        method: str = 'standard',
        columns: Optional[List[str]] = None
    ):
        """
        Initialize FeatureNormalizer.
        
        Args:
            method: Normalization method
            columns: Columns to normalize (None = all numeric)
        """
        self.method = method
        self.columns = columns
        self._scaler = None
        self._numeric_columns = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit normalizer to data."""
        X = pd.DataFrame(X)
        
        # Identify numeric columns
        if self.columns is None:
            self._numeric_columns = X.select_dtypes(
                include=[np.number]
            ).columns.tolist()
        else:
            self._numeric_columns = [
                c for c in self.columns if c in X.columns
            ]
        
        if not self._numeric_columns or self.method == 'none':
            return self
        
        # Create and fit scaler
        if self.method == 'standard':
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()
        elif self.method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self._scaler = MinMaxScaler()
        elif self.method == 'robust':
            from sklearn.preprocessing import RobustScaler
            self._scaler = RobustScaler()
        elif self.method == 'log':
            # Log transform doesn't need fitting
            pass
        
        if self._scaler is not None:
            self._scaler.fit(X[self._numeric_columns])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted normalization."""
        X = pd.DataFrame(X).copy()
        
        if not self._numeric_columns or self.method == 'none':
            return X
        
        if self.method == 'log':
            # Add small constant to avoid log(0)
            for col in self._numeric_columns:
                min_val = X[col].min()
                offset = abs(min_val) + 1 if min_val <= 0 else 0
                X[col] = np.log(X[col] + offset)
        elif self._scaler is not None:
            X[self._numeric_columns] = self._scaler.transform(
                X[self._numeric_columns]
            )
        
        return X
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse the normalization."""
        X = pd.DataFrame(X).copy()
        
        if self._scaler is not None:
            X[self._numeric_columns] = self._scaler.inverse_transform(
                X[self._numeric_columns]
            )
        
        return X
