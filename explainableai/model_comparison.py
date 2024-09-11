# model_comparison.py
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def compare_models(models, X, y, cv=5):
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        results[name] = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
        }
    return results

def plot_roc_curves(models, X, y):
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        auc = roc_auc_score(y, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Model Comparison')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves_comparison.png')
    plt.close()

def mcnemar_test(model1, model2, X, y):
    y_pred1 = model1.predict(X)
    y_pred2 = model2.predict(X)
    
    contingency_table = np.zeros((2, 2))
    contingency_table[0, 0] = np.sum((y_pred1 == y) & (y_pred2 == y))
    contingency_table[0, 1] = np.sum((y_pred1 == y) & (y_pred2 != y))
    contingency_table[1, 0] = np.sum((y_pred1 != y) & (y_pred2 == y))
    contingency_table[1, 1] = np.sum((y_pred1 != y) & (y_pred2 != y))
    
    statistic, p_value = stats.mcnemar(contingency_table, correction=True)
    return statistic, p_value