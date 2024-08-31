import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

def plot_feature_importance(feature_importance):
    plt.figure(figsize=(10, 6))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features, importance = zip(*sorted_features)
    plt.bar(features, importance)
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_partial_dependence(model, X, feature_importance):
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
    top_feature_names = [feature for feature, _ in top_features]
    
    fig, ax = plt.subplots(figsize=(12, 4))
    display = PartialDependenceDisplay.from_estimator(
        model, X, top_feature_names, 
        kind="average", subsample=1000, 
        n_jobs=3, grid_resolution=20
    )
    display.plot(ax=ax)
    plt.tight_layout()
    plt.show()