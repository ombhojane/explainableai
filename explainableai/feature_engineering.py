from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

def create_keras_model(input_dim):
    # Create a Keras model for binary classification
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_keras_model(X, y):
    model = create_keras_model(X.shape[1])
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    return model

def predict_with_keras(model, X):
    predictions = model.predict(X)
    return (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions

def automated_feature_engineering(X_train, X_test=None):
    # Convert categorical variables to one-hot encoding
    X_train_encoded = pd.get_dummies(X_train, drop_first=True)
    
    if X_test is not None:
        X_test_encoded = pd.get_dummies(X_test, drop_first=True)
        # Ensure X_test has the same columns as X_train
        for col in X_train_encoded.columns:
            if col not in X_test_encoded.columns:
                X_test_encoded[col] = 0
        X_test_encoded = X_test_encoded[X_train_encoded.columns]
    
    feature_names = X_train_encoded.columns.tolist()
    
    if X_test is not None:
        return X_train_encoded.values, X_test_encoded.values, feature_names
    
    return X_train_encoded.values, feature_names

# Example usage:
# X_train = ...  # your training feature data as a DataFrame
# X_test = ...   # your testing feature data as a DataFrame (optional)
# y_train = ...  # your training labels as a numpy array

# X_train_encoded, X_test_encoded, feature_names = automated_feature_engineering(X_train, X_test)
# model = train_keras_model(X_train_encoded, y_train)  # Train the Keras model
# predictions = predict_with_keras(model, X_test_encoded)  # Make predictions