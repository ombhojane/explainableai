# feature_analysis.py
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

def calculate_shap_values(model, X, feature_names):
    try:
        # Convert X to a DataFrame if it's not already
        X = pd.DataFrame(X, columns=feature_names)
        
        if hasattr(model, "predict_proba"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification, we take the positive class
        else:
            explainer = shap.KernelExplainer(model.predict, X)
            shap_values = explainer.shap_values(X)

        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.tight_layout()
        plt.show()
        plt.close()
        
        return shap_values
    except Exception as e:
        print(f"Error calculating SHAP values: {e}")
        print("Model type:", type(model))
        print("X shape:", X.shape)
        print("X dtype:", X.dtypes)
        print("Feature names:", feature_names)
        return None

# Example usage:
# X = ...  # your feature data as a numpy array
# y = ...  # your labels as a numpy array

# model = train_keras_model(X, y)  # Train the Keras model
# shap_values = calculate_shap_values(model, X, feature_names)  # Calculate SHAP values