# explainableai/core.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from .visualizations import (
    plot_feature_importance, plot_partial_dependence, plot_learning_curve,
    plot_roc_curve, plot_precision_recall_curve, plot_correlation_heatmap
)
from .model_evaluation import evaluate_model, cross_validate
from .feature_analysis import calculate_shap_values
from .feature_interaction import analyze_feature_interactions
from .llm_explanations import initialize_gemini, get_llm_explanation, get_prediction_explanation

class XAIWrapper:
    def __init__(self):
        self.model = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.is_classifier = None
        self.scaler = None
        self.gemini_model = initialize_gemini()

    def fit(self, model, X, y, feature_names=None):
        self.model = model
        self.X = X
        self.y = y
        self.feature_names = feature_names if feature_names is not None else [f"feature_{i}" for i in range(X.shape[1])]
        self.is_classifier = hasattr(model, "predict_proba")

        # Preprocess the data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        print("Fitting model and analyzing...")
        self.model.fit(X_scaled, y)
        return self

    def analyze(self):
        results = {}

        print("Evaluating model performance...")
        results['model_performance'] = evaluate_model(self.model, self.X, self.y, self.is_classifier)

        print("Calculating feature importance...")
        results['feature_importance'] = self._calculate_feature_importance()

        print("Generating visualizations...")
        self._generate_visualizations(results['feature_importance'])

        print("Calculating SHAP values...")
        results['shap_values'] = calculate_shap_values(self.model, self.X, self.feature_names)


        print("Performing cross-validation...")
        mean_score, std_score = cross_validate(self.model, self.X, self.y)
        results['cv_scores'] = (mean_score, std_score)

        self._print_results(results)

        print("Generating LLM explanation...")
        results['llm_explanation'] = get_llm_explanation(self.gemini_model, results)

        return results

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been fitted. Please run fit() first.")
        
        X_scaled = self.scaler.transform(X)
        
        if self.is_classifier:
            prediction = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            return prediction, probabilities
        else:
            prediction = self.model.predict(X_scaled)
            return prediction

    def explain_prediction(self, input_data):
        input_df = pd.DataFrame([input_data])
        prediction, probabilities = self.predict(input_df)
        explanation = get_prediction_explanation(self.gemini_model, input_data, prediction[0], probabilities[0], self.feature_importance)
        return prediction[0], probabilities[0], explanation
    
    def _calculate_feature_importance(self):
        perm_importance = permutation_importance(self.model, self.X, self.y, n_repeats=10, random_state=42)
        feature_importance = {feature: importance for feature, importance in zip(self.feature_names, perm_importance.importances_mean)}
        return dict(sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True))

    def _generate_visualizations(self, feature_importance):
        plot_feature_importance(feature_importance)
        plot_partial_dependence(self.model, self.X, feature_importance, self.feature_names)
        plot_learning_curve(self.model, self.X, self.y)
        plot_correlation_heatmap(pd.DataFrame(self.X, columns=self.feature_names))
        if self.is_classifier:
            plot_roc_curve(self.model, self.X, self.y)
            plot_precision_recall_curve(self.model, self.X, self.y)

    def _print_results(self, results):
        print("\nModel Performance:")
        for metric, value in results['model_performance'].items():
            if isinstance(value, (int, float, np.float64)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}:\n{value}")

        print("\nTop 5 Important Features:")
        for feature, importance in list(results['feature_importance'].items())[:5]:
            print(f"{feature}: {importance:.4f}")

        print(f"\nCross-validation Score: {results['cv_scores'][0]:.4f} (+/- {results['cv_scores'][1]:.4f})")

        print("\nVisualizations saved:")
        print("- Feature Importance: feature_importance.png")
        print("- Partial Dependence: partial_dependence.png")
        print("- Learning Curve: learning_curve.png")
        print("- Correlation Heatmap: correlation_heatmap.png")
        if self.is_classifier:
            print("- ROC Curve: roc_curve.png")
            print("- Precision-Recall Curve: precision_recall_curve.png")

        if results['shap_values'] is not None:
            print("\nSHAP values calculated successfully. See 'shap_summary.png' for visualization.")
        else:
            print("\nSHAP values calculation failed. Please check the console output for more details.")
        
    @staticmethod
    def perform_eda(df):
        print("\nExploratory Data Analysis:")
        print(f"Dataset shape: {df.shape}")
        print("\nDataset info:")
        df.info()
        print("\nSummary statistics:")
        print(df.describe())
        print("\nMissing values:")
        print(df.isnull().sum())
        print("\nData types:")
        print(df.dtypes)
        print("\nUnique values in target column:")
        print(df[df.columns[-1]].value_counts())