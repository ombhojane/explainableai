import colorama
from colorama import Fore, Style
from tensorflow import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .visualizations import (
    plot_feature_importance, plot_partial_dependence, plot_learning_curve,
    plot_roc_curve, plot_precision_recall_curve, plot_correlation_heatmap
)
from .model_evaluation import evaluate_model, cross_validate
from .feature_analysis import calculate_shap_values
from .feature_interaction import analyze_feature_interactions
from .llm_explanations import initialize_gemini, get_llm_explanation, get_prediction_explanation
from .report_generator import ReportGenerator
from .model_selection import compare_models
from reportlab.platypus import PageBreak

class XAIWrapper:
    def __init__(self):
        self.model = None
        self.keras_model = None  # Add Keras model attribute
        self.X = None
        self.y = None
        self.feature_names = None
        self.is_classifier = None
        self.preprocessor = None
        self.label_encoder = None
        self.categorical_columns = None
        self.numerical_columns = None
        self.gemini_model = initialize_gemini()
        self.feature_importance = None
        self.results = None  # Add this line to store analysis results

    # ... existing methods ...

    def create_keras_model(self, input_dim):
        # Create a Keras model for anomaly detection
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=input_dim))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def fit_keras_model(self, X, y):
        self.keras_model = self.create_keras_model(X.shape[1])
        self.keras_model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    def predict_keras(self, X):
        if self.keras_model is None:
            raise ValueError("Keras model has not been fitted. Please run fit_keras_model() first.")
        
        X = self._preprocess_input(X)
        prediction = self.keras_model.predict(X)
        return prediction

    # ... existing methods ...

    def analyze(self):
        results = {}

        print("Evaluating model performance...")
        results['model_performance'] = evaluate_model(self.model, self.X, self.y, self.is_classifier)

        print("Calculating feature importance...")
        self.feature_importance = self._calculate_feature_importance()
        results['feature_importance'] = self.feature_importance

        print("Generating visualizations...")
        self._generate_visualizations(self.feature_importance)

        print("Calculating SHAP values...")
        results['shap_values'] = calculate_shap_values(self.model, self.X, self.feature_names)

        print("Performing cross-validation...")
        mean_score, std_score = cross_validate(self.model, self.X, self.y)
        results['cv_scores'] = (mean_score, std_score)

        print("Model comparison results:")
        results['model_comparison'] = self.model_comparison_results

        self._print_results(results)

        print("Generating LLM explanation...")
        results['llm_explanation'] = get_llm_explanation(self.gemini_model, results)

        self.results = results
        return results

    # ... existing methods ...

# Example usage:
# xai_wrapper = XAIWrapper()
# xai_wrapper.fit(models, X, y)
# xai_wrapper.fit_keras_model(X, y)  # Fit the Keras model
# predictions = xai_wrapper.predict_keras(X)  # Make predictions with the Keras model