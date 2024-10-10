# model_interpretability.py
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import logging

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def calculate_shap_values(model, X):
    logger.debug("Calculating values...")
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        logger.info("Values caluated...")
        return shap_values
    except Exception as e:
        logger.error(f"Some error occurred in calculating values...{str(e)}")

def plot_shap_summary(shap_values, X):
    logger.debug("Summary...")
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig('shap_summary.png')
        plt.close()
    except TypeError as e:
        logger.error(f"Error in generating SHAP summary plot: {str(e)}")
        logger.error("Attempting alternative SHAP visualization...")
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values.values, X.values, feature_names=X.columns.tolist(), plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig('shap_summary.png')
            plt.close()
        except Exception as e2:
            logger.error(f"Alternative SHAP visualization also failed: {str(e2)}")
            logger.error("Skipping SHAP summary plot.")

def get_lime_explanation(model, X, instance, feature_names):
    logger.debug("Explaining model...")
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X,
            feature_names=feature_names,
            class_names=['Negative', 'Positive'],
            mode='classification'
        )
        exp = explainer.explain_instance(instance, model.predict_proba)
        logger.info("Model explained...")
        return exp
    except Exception as e:
        logger.error(f"Some error occurred in explaining model...{str(e)}")

def plot_lime_explanation(exp):
    exp.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig('lime_explanation.png')
    plt.close()

def plot_ice_curve(model, X, feature, num_ice_lines=50):
    ice_data = X.copy()
    feature_values = np.linspace(X[feature].min(), X[feature].max(), num=100)
    
    plt.figure(figsize=(10, 6))
    for _ in range(num_ice_lines):
        ice_instance = ice_data.sample(n=1, replace=True)
        predictions = []
        for value in feature_values:
            ice_instance[feature] = value
            predictions.append(model.predict_proba(ice_instance)[0][1])
        plt.plot(feature_values, predictions, color='blue', alpha=0.1)
    
    plt.xlabel(feature)
    plt.ylabel('Predicted Probability')
    plt.title(f'ICE Plot for {feature}')
    plt.tight_layout()
    plt.savefig(f'ice_plot_{feature}.png')
    plt.close()