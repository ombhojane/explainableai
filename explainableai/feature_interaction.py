# feature_interaction.py
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence
import time

def analyze_feature_interactions(model, X, feature_names, top_n=5, max_interactions=10):
    print("Starting feature interaction analysis...")
    feature_importance = dict(zip(feature_names, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_feature_names = [f[0] for f in top_features]

    interactions = []
    for i, (f1, f2) in enumerate(itertools.combinations(top_feature_names, 2)):
        if i >= max_interactions:
            print(f"Reached maximum number of interactions ({max_interactions}). Stopping analysis.")
            break
        
        print(f"Analyzing interaction between {f1} and {f2}...")
        start_time = time.time()
        f1_idx = feature_names.index(f1)
        f2_idx = feature_names.index(f2)
        pd_result = partial_dependence(model, X, features=[f1_idx, f2_idx], kind="average")
        interactions.append((f1, f2, pd_result))
        print(f"Interaction analysis for {f1} and {f2} completed in {time.time() - start_time:.2f} seconds.")

    for i, (f1, f2, (pd_values, (ax1_values, ax2_values))) in enumerate(interactions):
        print(f"Plotting interaction {i+1} between {f1} and {f2}...")
        fig, ax = plt.subplots(figsize=(10, 6))
        XX, YY = np.meshgrid(ax1_values, ax2_values)
        Z = pd_values.reshape(XX.shape).T
        contour = ax.contourf(XX, YY, Z, cmap="RdBu_r", alpha=0.5)
        ax.set_xlabel(f1)
        ax.set_ylabel(f2)
        ax.set_title(f'Partial Dependence of {f1} and {f2}')
        plt.colorbar(contour)
        plt.savefig(f'interaction_{i+1}_{f1}_{f2}.png')
        plt.close()

    print("Feature interaction analysis completed.")
    return interactions