# explainable_ai/model_evaluation.py

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np

def evaluate_model(model, X, y, is_classifier):
    if is_classifier:
        return evaluate_classifier(model, X, y)
    else:
        return evaluate_regressor(model, X, y)

def evaluate_classifier(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y, y_pred)
    class_report = classification_report(y, y_pred)
    
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report
    }

def evaluate_regressor(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    return {
        "mean_squared_error": mse,
        "r2_score": r2
    }

def cross_validate(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean(), scores.std()