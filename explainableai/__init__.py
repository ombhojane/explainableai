from sklearn.ensemble import IsolationForest
from tensorflow import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
import numpy as np

class AnomalyDetector:
    def __init__(self, X):
        # Validate input data
        if not isinstance(X, np.ndarray):
            raise ValueError("Input data X must be a numpy array.")
        if X.ndim != 2:
            raise ValueError("Input data X must be a 2D array.")
        
        self.X = X
        self.iso_forest = IsolationForest(contamination=0.1, random_state=42)
        self.keras_model = self.create_keras_model()

    def create_keras_model(self):
        # Create a Keras model for anomaly detection
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=self.X.shape[1]))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def detect_anomalies(self):
        # Fit the Isolation Forest model and predict anomalies
        anomalies = self.iso_forest.fit_predict(self.X)
        # Convert -1 (anomaly) and 1 (normal) to 1 and 0
        return (anomalies == -1).astype(int)

    def train_keras_model(self, y):
        # Ensure the model is compiled before training
        if not self.keras_model:
            raise ValueError("Keras model is not created.")
        
        # Fit the Keras model on the data and labels
        self.keras_model.fit(self.X, y, epochs=10, batch_size=32, verbose=0)

# Example usage:
# X = ...  # your data as a numpy array
# anomaly_detector = AnomalyDetector(X)
# anomalies = anomaly_detector.detect_anomalies()
# y = ...  # your labels as a numpy array
# anomaly_detector.train_keras_model(y)