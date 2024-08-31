import time
from .utils import explain_model, calculate_metrics
from .visualizations import plot_feature_importance, plot_partial_dependence
from .feature_analysis import calculate_shap_values

def xai_wrapper(train_function):
    def wrapper(*args, **kwargs):
        print("Starting XAI wrapper...")
        
        X_train, y_train = args[0], args[1]
        X_test, y_test = kwargs.get('X_test'), kwargs.get('y_test')
        
        start_time = time.time()
        model = train_function(X_train, y_train)
        end_time = time.time()
        
        training_time = end_time - start_time
        print(f"Model training completed in {training_time:.2f} seconds.")
        
        explanation = explain_model(model, X_train, y_train, X_test, y_test)
        metrics = calculate_metrics(model, X_test, y_test)
        
        # Generate visualizations
        plot_feature_importance(explanation['feature_importance'])
        plot_partial_dependence(model, X_train, explanation['feature_importance'])
        
        # Calculate SHAP values
        shap_values = calculate_shap_values(model, X_train)
        
        return model, {
            'explanation': explanation,
            'metrics': metrics,
            'shap_values': shap_values
        }
    
    return wrapper