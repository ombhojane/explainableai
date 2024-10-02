# model_interpretability.py
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np

def calculate_shap_values(model, X, feature_names, model_type):
    if model_type == 'tensorflow':
        # Use DeepExplainer for TensorFlow models
        background = X[np.random.choice(X.shape[0], 100, replace=False)]
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X)
        # For classifiers, shap_values is a list
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        return shap.Explanation(values=shap_values, data=X, feature_names=feature_names)
    else:
        # Original SHAP calculation for scikit-learn models
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        return shap_values

def plot_shap_summary(shap_values, X):
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig('shap_summary.png')
        plt.close()
    except TypeError as e:
        print(f"Error in generating SHAP summary plot: {e}")
        print("Attempting alternative SHAP visualization...")
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values.values, X.values, feature_names=X.columns.tolist(), plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig('shap_summary.png')
            plt.close()
        except Exception as e2:
            print(f"Alternative SHAP visualization also failed: {e2}")
            print("Skipping SHAP summary plot.")




def get_lime_explanation(model, X, instance, feature_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X,
        feature_names=feature_names,
        class_names=['Negative', 'Positive'],
        mode='classification'
    )
    exp = explainer.explain_instance(instance, model.predict_proba)
    return exp

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
