from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def explain_model(model, X_train, y_train, X_test, y_test):
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    feature_importance = {f"feature_{i}": importance for i, importance in enumerate(result.importances_mean)}
    
    return {
        "feature_importance": feature_importance,
        "model_type": str(type(model)),
    }

def calculate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        "mse": mse,
        "r2": r2,
    }