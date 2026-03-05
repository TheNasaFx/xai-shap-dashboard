"""
Data Processor - Main Data Processing Pipeline
===============================================

Comprehensive data preprocessing pipeline for machine learning,
with special attention to preparing data for explainability analysis.

Author: XAI-SHAP Framework
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Main data processing class for XAI framework.
    
    This class handles the complete data preprocessing pipeline:
    1. Data validation and cleaning
    2. Missing value handling
    3. Categorical variable encoding
    4. Feature normalization/scaling
    5. Train/test splitting
    
    The processor is designed to maintain feature interpretability,
    which is crucial for generating meaningful SHAP explanations.
    
    Attributes:
        config: Configuration manager instance
        scalers: Dictionary of fitted scalers
        encoders: Dictionary of fitted encoders
        feature_names: List of processed feature names
        
    Example:
        >>> processor = DataProcessor(config)
        >>> result = processor.process(df, target='label')
        >>> X_train, X_test = result['X_train'], result['X_test']
    """
    
    def __init__(self, config=None):
        """
        Initialize DataProcessor.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self._original_dtypes = {}
        self._categorical_mappings = {}
        
        # Get configuration settings
        if config:
            self.normalization_method = config.get(
                "data_processing.normalization_method", "standard"
            )
            self.missing_strategy = config.get(
                "data_processing.missing_value_strategy", "median"
            )
            self.encoding_method = config.get(
                "data_processing.categorical_encoding", "onehot"
            )
            test_size_val = config.get("data_processing.test_size", 0.2)
            self.test_size = test_size_val if 0 < test_size_val < 1 else 0.2
            self.random_state = config.get("data_processing.random_state", 42)
        else:
            self.normalization_method = "standard"
            self.missing_strategy = "median"
            self.encoding_method = "onehot"
            self.test_size = 0.2
            self.random_state = 42
    
    def process(
        self,
        data: pd.DataFrame,
        target: str,
        exclude_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Complete data processing pipeline.
        
        Args:
            data: Input DataFrame
            target: Name of target column
            exclude_columns: Columns to exclude from processing
            
        Returns:
            Dictionary containing:
            - X_train, X_test: Processed feature arrays
            - y_train, y_test: Target arrays
            - feature_names: List of feature names
            - processing_info: Metadata about processing
        """
        logger.info(f"Starting data processing: {data.shape[0]} rows, {data.shape[1]} columns")
        
        # Store original data info
        exclude_columns = exclude_columns or []
        self._original_dtypes = data.dtypes.to_dict()
        
        # Separate features and target
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        
        X = data.drop(columns=[target] + exclude_columns)
        y = data[target].copy()
        
        # Identify column types
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Numeric columns: {len(numeric_cols)}, Categorical: {len(categorical_cols)}")
        
        # Step 1: Handle missing values
        X = self._handle_missing_values(X, numeric_cols, categorical_cols)
        
        # Step 2: Encode categorical variables
        X, encoded_feature_names = self._encode_categorical(X, categorical_cols)
        
        # Step 3: Normalize numeric features
        X = self._normalize_features(X, numeric_cols)
        
        # Update feature names
        self.feature_names = numeric_cols + encoded_feature_names
        
        # Step 4: Handle target variable
        y, target_encoder = self._process_target(y)
        
        # Step 5: Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if self._is_classification(y) else None
        )
        
        # Convert to numpy arrays
        X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        y_train = y_train.values if isinstance(y_train, pd.Series) else y_train
        y_test = y_test.values if isinstance(y_test, pd.Series) else y_test
        
        processing_info = {
            'n_original_features': len(data.columns) - 1,
            'n_processed_features': len(self.feature_names),
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'normalization_method': self.normalization_method,
            'encoding_method': self.encoding_method,
            'missing_values_handled': True
        }
        
        logger.info(f"Processing complete: {len(self.feature_names)} features")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'processing_info': processing_info,
            'target_encoder': target_encoder
        }
    
    def _handle_missing_values(
        self,
        X: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: List[str]
    ) -> pd.DataFrame:
        """Handle missing values in data."""
        X = X.copy()
        
        # Numeric columns
        for col in numeric_cols:
            if X[col].isnull().any():
                if self.missing_strategy == "mean":
                    fill_value = X[col].mean()
                elif self.missing_strategy == "median":
                    fill_value = X[col].median()
                elif self.missing_strategy == "mode":
                    fill_value = X[col].mode()[0]
                else:  # drop
                    continue
                
                X[col] = X[col].fillna(fill_value)
                logger.debug(f"Filled {col} missing values with {self.missing_strategy}: {fill_value}")
        
        # Categorical columns
        for col in categorical_cols:
            if X[col].isnull().any():
                fill_value = X[col].mode()[0] if len(X[col].mode()) > 0 else "Unknown"
                X[col] = X[col].fillna(fill_value)
        
        # Drop rows with remaining missing values
        if X.isnull().any().any():
            n_before = len(X)
            X = X.dropna()
            logger.warning(f"Dropped {n_before - len(X)} rows with missing values")
        
        return X
    
    def _encode_categorical(
        self,
        X: pd.DataFrame,
        categorical_cols: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Encode categorical variables."""
        if not categorical_cols:
            return X, []
        
        X = X.copy()
        encoded_feature_names = []
        
        if self.encoding_method == "onehot":
            # One-hot encoding
            for col in categorical_cols:
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                encoded_feature_names.extend(dummies.columns.tolist())
                
                # Store mapping for interpretability
                self._categorical_mappings[col] = {
                    'method': 'onehot',
                    'categories': dummies.columns.tolist()
                }
                
        elif self.encoding_method == "label":
            # Label encoding
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.encoders[col] = le
                encoded_feature_names.append(col)
                
                self._categorical_mappings[col] = {
                    'method': 'label',
                    'classes': le.classes_.tolist()
                }
        
        return X, encoded_feature_names
    
    def _normalize_features(
        self,
        X: pd.DataFrame,
        numeric_cols: List[str]
    ) -> pd.DataFrame:
        """Normalize numeric features."""
        if not numeric_cols:
            return X
        
        X = X.copy()
        
        # Select scaler
        if self.normalization_method == "standard":
            scaler = StandardScaler()
        elif self.normalization_method == "minmax":
            scaler = MinMaxScaler()
        elif self.normalization_method == "robust":
            scaler = RobustScaler()
        else:
            return X  # No normalization
        
        # Fit and transform
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        self.scalers['numeric'] = scaler
        
        return X
    
    def _process_target(
        self,
        y: pd.Series
    ) -> Tuple[pd.Series, Optional[LabelEncoder]]:
        """Process target variable."""
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name=y.name)
            return y, le
        return y, None
    
    def _is_classification(self, y: Union[pd.Series, np.ndarray]) -> bool:
        """Determine if this is a classification task."""
        unique_values = np.unique(y)
        return len(unique_values) <= 20  # Heuristic threshold
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted processors.
        
        Args:
            X: New data to transform
            
        Returns:
            Transformed numpy array
        """
        X = X.copy()
        
        # Apply categorical encoding
        for col, mapping in self._categorical_mappings.items():
            if col in X.columns:
                if mapping['method'] == 'label':
                    X[col] = self.encoders[col].transform(X[col].astype(str))
                elif mapping['method'] == 'onehot':
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    # Align columns with training data
                    for cat_col in mapping['categories']:
                        if cat_col not in dummies.columns:
                            dummies[cat_col] = 0
                    dummies = dummies[mapping['categories']]
                    X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
        
        # Apply normalization
        if 'numeric' in self.scalers:
            numeric_cols = [c for c in X.columns if c in self._original_dtypes 
                          and np.issubdtype(self._original_dtypes[c], np.number)]
            if numeric_cols:
                X[numeric_cols] = self.scalers['numeric'].transform(X[numeric_cols])
        
        return X[self.feature_names].values
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about processed features."""
        return {
            'feature_names': self.feature_names,
            'categorical_mappings': self._categorical_mappings,
            'normalization_method': self.normalization_method,
            'n_features': len(self.feature_names)
        }
