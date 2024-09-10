# explainable_ai/core.py

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from .visualizations import plot_feature_importance, plot_partial_dependence, plot_learning_curve, plot_roc_curve, plot_precision_recall_curve
from .feature_analysis import calculate_shap_values
from .model_evaluation import evaluate_model
from .feature_interaction import analyze_feature_interactions

class XAIWrapper:
    def __init__(self):
        self.model = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.is_classifier = None

    def fit(self, model, X, y, feature_names=None):
        self.model = model
        self.X = X
        self.y = y
        self.feature_names = feature_names if feature_names is not None else [f"feature_{i}" for i in range(X.shape[1])]
        self.is_classifier = hasattr(model, "predict_proba")

        print("Fitting model and analyzing...")
        self.model.fit(X, y)
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

        print("Analyzing feature interactions...")
        results['interactions'] = analyze_feature_interactions(self.model, self.X, self.feature_names)

        print("Performing cross-validation...")
        results['cv_scores'] = cross_val_score(self.model, self.X, self.y, cv=5)

        self._print_results(results)
        return results

    def _calculate_feature_importance(self):
        perm_importance = permutation_importance(self.model, self.X, self.y, n_repeats=10, random_state=42)
        feature_importance = {feature: importance for feature, importance in zip(self.feature_names, perm_importance.importances_mean)}
        return dict(sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True))

    def _generate_visualizations(self, feature_importance):
        plot_feature_importance(feature_importance)
        plot_partial_dependence(self.model, self.X, feature_importance, self.feature_names)
        plot_learning_curve(self.model, self.X, self.y)
        if self.is_classifier:
            plot_roc_curve(self.model, self.X, self.y)
            plot_precision_recall_curve(self.model, self.X, self.y)

    def _print_results(self, results):
        print("\nModel Performance:")
        for metric, value in results['model_performance'].items():
            print(f"{metric}: {value:.4f}")

        print("\nTop 5 Important Features:")
        for feature, importance in list(results['feature_importance'].items())[:5]:
            print(f"{feature}: {importance:.4f}")

        print(f"\nCross-validation Score: {results['cv_scores'].mean():.4f} (+/- {results['cv_scores'].std() * 2:.4f})")

        print("\nVisualizations saved:")
        print("- Feature Importance: feature_importance.png")
        print("- Partial Dependence: partial_dependence.png")
        print("- Learning Curve: learning_curve.png")
        if self.is_classifier:
            print("- ROC Curve: roc_curve.png")
            print("- Precision-Recall Curve: precision_recall_curve.png")

        print("\nAnalyzing feature interactions...")
        for i, (f1, f2, _) in enumerate(results['interactions'][:5]):
            print(f"Interaction {i+1}: {f1} and {f2}")

        if results['shap_values'] is not None:
            print("\nSHAP values calculated successfully. See 'shap_summary.png' for visualization.")
        else:
            print("\nSHAP values calculation failed. Please check the console output for more details.")

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been fitted. Please run fit() first.")
        
        if self.is_classifier:
            prediction = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            return prediction, probabilities
        else:
            prediction = self.model.predict(X)
            return prediction