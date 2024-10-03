# feature_analysis.py
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
def calculate_shap_values(model, X, feature_names):
    try:
        # Convert X to a DataFrame if it's not already
        logger.debug("Convert X to Dataframe...")
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
        logger.info("Dataframe Created...")
        return shap_values
    except Exception as e:
        logger.error(f"Error calculating SHAP values: {e}")
        logger.error("Model type:", type(model))
        logger.error("X shape:", X.shape)
        logger.error("X dtype:", X.dtypes)
        logger.error("Feature names:", feature_names)
        return None