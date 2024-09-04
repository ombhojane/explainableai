# feature_analysis.py
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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