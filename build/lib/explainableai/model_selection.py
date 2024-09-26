# model_selection.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def get_default_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }

def compare_models(X_train, y_train, X_test, y_test, models=None):
    if models is None:
        models = get_default_models()
    
    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        model.fit(X_train, y_train)
        test_score = model.score(X_test, y_test)
        results[name] = {
            'cv_score': cv_scores.mean(),
            'test_score': test_score
        }
    
    plot_roc_curves(models, X_test, y_test)
    
    return results

def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('model_comparison_roc_curves.png')
    plt.close()