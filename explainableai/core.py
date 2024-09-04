# explainable_ai/core.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from .utils import explain_model, calculate_metrics
from .visualizations import plot_feature_importance, plot_partial_dependence, plot_interactive_feature_importance
from .feature_analysis import calculate_shap_values
from .feature_engineering import automated_feature_engineering
from .model_selection import compare_models
from .anomaly_detection import detect_anomalies
from .feature_selection import select_features
from .model_evaluation import cross_validate, plot_learning_curve, plot_roc_curve, plot_precision_recall_curve

class XAIWrapper:
    def __init__(self):
        self.model = None
        self.xai_results = None
        self.feature_names = None

    def analyze(self, data, target_column, test_size=0.2, random_state=42):
        print("Starting XAI analysis...")
        
        # Analyze dataset
        self._analyze_dataset(data)
        
        # Preprocess data
        X, y = self._preprocess_data(data, target_column)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Feature engineering
        X_train_fe, X_test_fe, feature_names = automated_feature_engineering(X_train, X_test)
        
        # Feature selection
        X_train_selected, selected_indices = select_features(X_train_fe, y_train)
        X_test_selected = X_test_fe[:, selected_indices]
        self.feature_names = [feature_names[i] for i in selected_indices]
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.model.fit(X_train_selected, y_train)
        
        # Explain model
        explanation = explain_model(self.model, X_train_selected, y_train, X_test_selected, y_test, self.feature_names)
        
        # Calculate metrics
        metrics = calculate_metrics(self.model, X_test_selected, y_test)
        
        # Generate visualizations
        plot_feature_importance(explanation['feature_importance'])
        plot_partial_dependence(self.model, X_train_selected, explanation['feature_importance'], self.feature_names)
        plot_interactive_feature_importance(explanation['feature_importance'])
        
        # Calculate SHAP values
        shap_values = calculate_shap_values(self.model, X_train_selected, self.feature_names)
        
        # Compare models
        model_comparison = compare_models(X_train_selected, y_train, X_test_selected, y_test)
        
        # Detect anomalies
        anomalies = detect_anomalies(X_test_selected)
        
        # Cross-validation
        cv_score, cv_std = cross_validate(self.model, X_train_selected, y_train)
        
        # Generate evaluation plots
        plot_learning_curve(self.model, X_train_selected, y_train)
        plot_roc_curve(self.model, X_test_selected, y_test)
        plot_precision_recall_curve(self.model, X_test_selected, y_test)
                
        self.xai_results = {
            'explanation': explanation,
            'metrics': metrics,
            'shap_values': shap_values,
            'model_comparison': model_comparison,
            'anomalies': anomalies,
            'cv_score': cv_score,
            'cv_std': cv_std,
        }
        
        print("XAI analysis completed.")
        self._print_results()

    def predict(self, input_data):
        if self.model is None:
            raise ValueError("Model has not been trained. Please run analyze() first.")
        
        input_df = pd.DataFrame([input_data])
        prediction = self.model.predict(input_df)[0]
        probabilities = self.model.predict_proba(input_df)[0]
        
        return prediction, probabilities[1]

    def _analyze_dataset(self, data):
        print("Dataset shape:", data.shape)
        print("\nSample data:")
        print(data.head())
        print("\nData types:")
        print(data.dtypes)
        print("\nSummary statistics:")
        print(data.describe())

    def _preprocess_data(self, data, target_column):
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X = pd.get_dummies(X, drop_first=True)
        return X, y

    def _print_results(self):
        print("\nModel Metrics:", self.xai_results['metrics'])
        print("\nTop 5 Important Features:")
        top_features = sorted(self.xai_results['explanation']['feature_importance'].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for feature, importance in top_features:
            print(f"{feature}: {importance:.4f}")

        print(f"\nCross-validation Score: {self.xai_results['cv_score']:.4f} (+/- {self.xai_results['cv_std']:.4f})")

        print("\nModel Comparison:")
        for model_name, scores in self.xai_results['model_comparison'].items():
            print(f"{model_name}: CV Score = {scores['cv_score']:.4f}, Test Score = {scores['test_score']:.4f}")

        print(f"\nAnomalies detected: {sum(self.xai_results['anomalies'] == -1)}")

        if self.xai_results['shap_values'] is not None:
            print("\nSHAP values calculated successfully. See 'shap_summary.png' for visualization.")
        else:
            print("\nSHAP values calculation failed. Please check the console output for more details.")

        print("\nVisualizations saved:")
        print("- Feature Importance: feature_importance.png")
        print("- Partial Dependence: partial_dependence.png")
        print("- Learning Curve: learning_curve.png")
        print("- ROC Curve: roc_curve.png")
        print("- Precision-Recall Curve: precision_recall_curve.png")
        print("- Model Comparison ROC Curves: model_comparison_roc_curves.png")
