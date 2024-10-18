import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
from .logging_config import logger
from .exceptions import ExplainableAIError

def calculate_shap_values(model, X, feature_names):
    logger.debug("Calculating SHAP values...")
    try:
        X_df = pd.DataFrame(X, columns=feature_names)
        if hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_df)
        else:
            explainer = shap.Explainer(model, X_df)
            shap_values = explainer(X_df)
        logger.info("SHAP values calculated successfully.")
        return shap_values
    except Exception as e:
        logger.error(f"Error in calculate_shap_values: {str(e)}")
        raise ExplainableAIError(f"Error in calculate_shap_values: {str(e)}")

def plot_shap_summary(shap_values, X, feature_names):
    logger.debug("Plotting SHAP summary...")
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, plot_type="bar", feature_names=feature_names, show=False)
        plt.tight_layout()
        
        # Save plot to file
        plt.savefig('shap_summary.png')
        logger.info("SHAP summary plot saved as 'shap_summary.png'")
        
        # Convert plot to base64 for display
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url
    except Exception as e:
        logger.error(f"Error in plot_shap_summary: {str(e)}")
        raise ExplainableAIError(f"Error in plot_shap_summary: {str(e)}")

def get_lime_explanation(model, X, instance, feature_names):
    logger.debug("Generating LIME explanation...")
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X,
            feature_names=feature_names,
            class_names=['Negative', 'Positive'],
            mode='classification' if hasattr(model, 'predict_proba') else 'regression'
        )
        exp = explainer.explain_instance(
            instance, 
            model.predict_proba if hasattr(model, 'predict_proba') else model.predict
        )
        logger.info("LIME explanation generated successfully.")
        return exp
    except Exception as e:
        logger.error(f"Error in get_lime_explanation: {str(e)}")
        raise ExplainableAIError(f"Error in get_lime_explanation: {str(e)}")

def plot_lime_explanation(exp):
    logger.debug("Plotting LIME explanation...")
    try:
        plt.figure(figsize=(12, 8))
        exp.as_pyplot_figure()
        plt.tight_layout()
        
        # Save plot to file
        plt.savefig('lime_explanation.png')
        logger.info("LIME explanation plot saved as 'lime_explanation.png'")
        
        # Convert plot to base64 for display
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url
    except Exception as e:
        logger.error(f"Error in plot_lime_explanation: {str(e)}")
        raise ExplainableAIError(f"Error in plot_lime_explanation: {str(e)}")

def interpret_model(model, X, feature_names, instance_index=0):
    logger.info("Starting model interpretation...")
    try:
        # SHAP analysis
        shap_values = calculate_shap_values(model, X, feature_names)
        shap_plot_url = plot_shap_summary(shap_values, X, feature_names)
        
        # LIME analysis
        instance = X[instance_index]
        lime_exp = get_lime_explanation(model, X, instance, feature_names)
        lime_plot_url = plot_lime_explanation(lime_exp)
        
        interpretation_results = {
            "shap_values": shap_values,
            "shap_plot_url": shap_plot_url,
            "lime_explanation": lime_exp,
            "lime_plot_url": lime_plot_url
        }
        
        logger.info("Model interpretation completed successfully.")
        return interpretation_results
    except Exception as e:
        logger.error(f"Error in interpret_model: {str(e)}")
        raise ExplainableAIError(f"Error in interpret_model: {str(e)}")