"""
Neural Network Model Wrapper - Нейрон сүлжээний загварын боодол
================================================================

Нейрон сүлжээний загваруудад зориулсан нэгдсэн интерфейстэй боодол.
sklearn MLP болон гүн сургалтын framework-уудыг дэмждэг.

Зохиогч: XAI-SHAP Framework
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np

from src.models.base_model import BaseModel


class NeuralNetworkModel(BaseModel):
    """
    Нейрон сүлжээний загварын боодол - нэгдсэн интерфейстэй.
    
    Энэ боодол дэмждэг:
    - sklearn MLPClassifier/Regressor (үндсэн)
    - TensorFlow/Keras загварууд (нэмэлт)
    - PyTorch загварууд (нэмэлт)
    
    Жишээ:
        >>> model = NeuralNetworkModel({
        ...     'hidden_layers': [128, 64, 32],
        ...     'activation': 'relu',
        ...     'learning_rate': 0.001
        ... })
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        framework: str = 'sklearn'
    ):
        """
        Нейрон сүлжээний загварыг эхлүүлэх.
        
        Параметрүүд:
            config: Загварын тохиргоо
            framework: Backend framework ('sklearn', 'tensorflow', 'pytorch')
        """
        super().__init__(config)
        self.framework = framework
        self._task_type = None
        self._keras_model = False
        self._pytorch_model = False
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[tuple] = None,
        **kwargs
    ) -> 'NeuralNetworkModel':
        """
        Нейрон сүлжээний загварыг сургалтын өгөгдөлд тохируулах.
        
        Параметрүүд:
            X: Сургалтын шинж чанарууд (features)
            y: Сургалтын шошго (labels)
            validation_data: (X_val, y_val) баталгаажуулалтын өгөгдөл
            **kwargs: Нэмэлт параметрүүд
            
        Буцаах:
            self
        """
        # Даалгаврын төрлийг тодорхойлох
        n_classes = len(np.unique(y))
        self._task_type = 'classification' if n_classes <= 20 else 'regression'
        
        if self.framework == 'sklearn':
            self._fit_sklearn(X, y, validation_data)
        elif self.framework == 'tensorflow':
            self._fit_tensorflow(X, y, validation_data)
        elif self.framework == 'pytorch':
            self._fit_pytorch(X, y, validation_data)
        else:
            raise ValueError(f"Тодорхойгүй framework: {self.framework}")
        
        self._is_fitted = True
        return self
    
    def _fit_sklearn(self, X, y, validation_data):
        """sklearn MLPClassifier/Regressor ашиглан сургах."""
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        
        # Hidden layers задлах
        hidden_layers = self.config.get('hidden_layers', [128, 64, 32])
        if isinstance(hidden_layers, list):
            hidden_layers = tuple(hidden_layers)
        
        params = {
            'hidden_layer_sizes': hidden_layers,
            'activation': self.config.get('activation', 'relu'),
            'learning_rate_init': self.config.get('learning_rate', 0.001),
            'max_iter': self.config.get('epochs', 200),
            'random_state': self.config.get('random_state', 42),
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10,
            'verbose': False
        }
        
        if self._task_type == 'classification':
            self._model = MLPClassifier(**params)
        else:
            self._model = MLPRegressor(**params)
        
        self._model.fit(X, y)
    
    def _fit_tensorflow(self, X, y, validation_data):
        """TensorFlow/Keras ашиглан сургах."""
        try:
            import tensorflow as tf
            from tensorflow import keras
            # Анхааруулгуудыг хаах
            tf.get_logger().setLevel('ERROR')
        except ImportError:
            raise ImportError("TensorFlow суулгаагүй байна. Дараах командыг ажиллуулна уу: pip install tensorflow")
        
        hidden_layers = self.config.get('hidden_layers', [128, 64, 32])
        activation = self.config.get('activation', 'relu')
        dropout_rate = self.config.get('dropout_rate', 0.3)
        learning_rate = self.config.get('learning_rate', 0.001)
        epochs = self.config.get('epochs', 100)
        batch_size = self.config.get('batch_size', 32)
        
        # Загварыг бүтээх
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(X.shape[1],)))
        
        for units in hidden_layers:
            model.add(keras.layers.Dense(units, activation=activation))
            model.add(keras.layers.Dropout(dropout_rate))
        
        # Гаралтын давхарга
        if self._task_type == 'classification':
            n_classes = len(np.unique(y))
            if n_classes == 2:
                model.add(keras.layers.Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'
            else:
                model.add(keras.layers.Dense(n_classes, activation='softmax'))
                loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        else:
            model.add(keras.layers.Dense(1))
            loss = 'mse'
            metrics = ['mae']
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        # Callback-ууд
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True,
                verbose=0
            )
        ]
        
        # Сургах
        model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0
        )
        
        self._model = model
        self._keras_model = True
    
    def _fit_pytorch(self, X, y, validation_data):
        """PyTorch ашиглан сургах."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch суулгаагүй байна. Дараах командыг ажиллуулна уу: pip install torch")
        
        hidden_layers = self.config.get('hidden_layers', [128, 64, 32])
        activation = self.config.get('activation', 'relu')
        dropout_rate = self.config.get('dropout_rate', 0.3)
        learning_rate = self.config.get('learning_rate', 0.001)
        epochs = self.config.get('epochs', 100)
        batch_size = self.config.get('batch_size', 32)
        
        # Загварын архитектурыг бүтээх
        layers = []
        input_dim = X.shape[1]
        
        for units in hidden_layers:
            layers.append(nn.Linear(input_dim, units))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = units
        
        # Гаралтын давхарга
        n_classes = len(np.unique(y))
        if self._task_type == 'classification':
            if n_classes == 2:
                layers.append(nn.Linear(input_dim, 1))
                layers.append(nn.Sigmoid())
                criterion = nn.BCELoss()
            else:
                layers.append(nn.Linear(input_dim, n_classes))
                criterion = nn.CrossEntropyLoss()
        else:
            layers.append(nn.Linear(input_dim, 1))
            criterion = nn.MSELoss()
        
        model = nn.Sequential(*layers)
        
        # Сургалт
        X_tensor = torch.FloatTensor(X)
        if self._task_type == 'classification' and n_classes == 2:
            y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        elif self._task_type == 'classification':
            y_tensor = torch.LongTensor(y)
        else:
            y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Сургалтын давталт
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        model.eval()
        self._model = model
        self._pytorch_model = True
        self._n_classes = n_classes
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Таамаглал хийх."""
        if not self._is_fitted:
            raise ValueError("Загвар сургагдаагүй байна. Эхлээд fit() дуудна уу.")
        
        if self.framework == 'sklearn':
            return self._model.predict(X)
        elif self.framework == 'tensorflow':
            predictions = self._model.predict(X, verbose=0)
            if self._task_type == 'classification':
                if predictions.shape[-1] == 1:
                    return (predictions > 0.5).astype(int).flatten()
                return np.argmax(predictions, axis=1)
            return predictions.flatten()
        elif self.framework == 'pytorch':
            import torch
            self._model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                predictions = self._model(X_tensor).numpy()
            if self._task_type == 'classification':
                if hasattr(self, '_n_classes') and self._n_classes == 2:
                    return (predictions > 0.5).astype(int).flatten()
                return np.argmax(predictions, axis=1)
            return predictions.flatten()
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Ангиллын магадлалыг таамаглах."""
        if not self._is_fitted or self._task_type != 'classification':
            return None
        
        if self.framework == 'sklearn':
            return self._model.predict_proba(X)
        elif self.framework == 'tensorflow':
            proba = self._model.predict(X, verbose=0)
            if proba.shape[-1] == 1:
                return np.column_stack([1 - proba.flatten(), proba.flatten()])
            return proba
        elif self.framework == 'pytorch':
            import torch
            self._model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                proba = self._model(X_tensor).numpy()
            if proba.shape[-1] == 1:
                return np.column_stack([1 - proba.flatten(), proba.flatten()])
            return proba
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Нейрон сүлжээнд built-in feature importance байдаггүй.
        Feature importance шинжилгээнд SHAP ашиглана уу.
        """
        return None
    
    @property
    def model(self):
        """Дотоод загварт хандах."""
        return self._model
