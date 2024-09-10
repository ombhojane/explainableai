# explainable_ai/visualizations.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def plot_feature_importance(feature_importance):
    plt.figure(figsize=(12, 8))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features, importance = zip(*sorted_features)
    plt.bar(features, importance)
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def plot_partial_dependence(model, X, feature_importance, feature_names):
    top_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    top_feature_indices = [feature_names.index(feature) for feature, _ in top_features]
    
    fig, ax = plt.subplots(figsize=(15, 5))
    display = PartialDependenceDisplay.from_estimator(
        model, X, top_feature_indices, 
        feature_names=feature_names,
        kind="average", subsample=1000, 
        n_jobs=3, grid_resolution=20
    )
    display.plot(ax=ax)
    
    for ax in display.axes_.ravel():
        ylim = ax.get_ylim()
        if ylim[0] == ylim[1]:
            ax.set_ylim(ylim[0] - 0.1, ylim[1] + 0.1)
    
    plt.suptitle('Partial Dependence of Top 3 Features')
    plt.tight_layout()
    plt.savefig('partial_dependence.png')
    plt.close()

def plot_learning_curve(model, X, y, cv=5):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    
    plt.figure(figsize=(10, 6))
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig('learning_curve.png')
    plt.close()

def plot_roc_curve(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

def plot_precision_recall_curve(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    average_precision = average_precision_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall curve: AP={average_precision:.2f}')
    plt.savefig('precision_recall_curve.png')
    plt.close()

def plot_interactive_feature_importance(feature_importance):
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features, importance = zip(*sorted_features)
    
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Bar(x=features, y=importance))
    fig.update_layout(
        title_text="Interactive Feature Importance",
        xaxis_title="Features",
        yaxis_title="Importance",
        height=600
    )
    fig.update_xaxes(tickangle=45)
    fig.write_html('interactive_feature_importance.html')

def plot_correlation_heatmap(X):
    corr = X.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Features')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()