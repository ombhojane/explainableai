# visualizations.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay
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
    plt.show()
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
    plt.show()
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
    plt.show()
    plt.close()