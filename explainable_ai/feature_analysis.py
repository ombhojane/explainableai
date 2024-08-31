import shap

def calculate_shap_values(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    
    shap.summary_plot(shap_values, X, plot_type="bar")
    
    return shap_values