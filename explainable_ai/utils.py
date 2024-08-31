from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.inspection import permutation_importance
import numpy as np

def explain_model(model, X_train, y_train, X_test, y_test, feature_names):
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    feature_importance = {feature: importance for feature, importance in zip(feature_names, result.importances_mean)}
    
    # Sort feature importance by absolute value
    feature_importance = dict(sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True))
    
    return {
        "feature_importance": feature_importance,
        "model_type": str(type(model)),
    }

def calculate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    if len(np.unique(y_test)) == 2:  # Binary classification
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average='weighted')
        }
    else:  # Regression or multi-class classification
        return {
            "mse": mean_squared_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred)
        }