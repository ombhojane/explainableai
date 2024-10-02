
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.inspection import permutation_importance
import shap


def calculate_metrics(model, X_test, y_test, model_type='sklearn'):
    if model_type == 'tensorflow':
        y_pred = model.predict(X_test)
        # Flatten predictions if necessary
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()
    else:
        y_pred = model.predict(X_test)
    
    if len(np.unique(y_test)) == 2:
        # Binary classification
        if model_type == 'tensorflow':
            y_pred = (y_pred > 0.5).astype(int)
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average='weighted')
        }
    elif y_pred.ndim > 1 and y_pred.shape[1] > 1:
        # Multi-class classification
        if model_type == 'tensorflow':
            y_pred = np.argmax(y_pred, axis=1)
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average='weighted')
        }
    else:
        # Regression
        return {
            "mse": mean_squared_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred)
        }

def explain_model(model, X_train, y_train, X_test, y_test, feature_names, model_type='sklearn'):
    if model_type == 'tensorflow':
        background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        feature_importance_values = np.mean(np.abs(shap_values), axis=0)
        feature_importance = {feature: importance for feature, importance in zip(feature_names, feature_importance_values)}
    else:
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        feature_importance = {feature: importance for feature, importance in zip(feature_names, result.importances_mean)}
    
    feature_importance = dict(sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True))
    return {
        "feature_importance": feature_importance,
        "model_type": str(type(model)),
    }

def explain_model(model, X_train, y_train, X_test, y_test, feature_names, model_type='sklearn'):
    if model_type == 'tensorflow':
        background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        feature_importance_values = np.mean(np.abs(shap_values), axis=0)
        feature_importance = {feature: importance for feature, importance in zip(feature_names, feature_importance_values)}
    else:
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        feature_importance = {feature: importance for feature, importance in zip(feature_names, result.importances_mean)}
    
    feature_importance = dict(sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True))
    return {
        "feature_importance": feature_importance,
        "model_type": str(type(model)),
    }
