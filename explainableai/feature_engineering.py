# feature_interaction.py
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence
import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def analyze_feature_interactions(model, X, feature_names, top_n=5, max_interactions=10):
    logger.debug("Starting feature interaction analysis...")
    try:
        # Ensure model has feature_importances_
        if not hasattr(model, 'feature_importances_'):
            raise AttributeError("Model does not have 'feature_importances_' attribute.")

        # Calculate and sort feature importances
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_feature_names = [f[0] for f in top_features]

        interactions = []
        for i, (f1, f2) in enumerate(itertools.combinations(top_feature_names, 2)):
            if i >= max_interactions:
                logger.info(f"Reached maximum number of interactions ({max_interactions}). Stopping analysis.")
                break

            logger.info(f"Analyzing interaction between {f1} and {f2}...")
            start_time = time.time()

            try:
                f1_idx = feature_names.index(f1)
                f2_idx = feature_names.index(f2)
            except ValueError as ve:
                logger.error(f"Feature {f1} or {f2} not found in feature_names: {ve}")
                continue

            try:
                pd_result = partial_dependence(model, X, features=[f1_idx, f2_idx], kind="average")
            except Exception as pd_error:
                logger.error(f"Partial dependence computation failed for {f1} and {f2}: {pd_error}")
                continue

            interactions.append((f1, f2, pd_result))
            logger.info(f"Interaction analysis for {f1} and {f2} completed in {time.time() - start_time:.2f} seconds.")

        for i, (f1, f2, (pd_values, (ax1_values, ax2_values))) in enumerate(interactions):
            try:
                logger.debug(f"Plotting interaction {i+1} between {f1} and {f2}...")
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
            except Exception as plot_error:
                logger.error(f"Failed to plot interaction for {f1} and {f2}: {plot_error}")

        logger.info("Feature interaction analysis completed.")
        return interactions

    except AttributeError as attr_err:
        logger.error(f"Model does not support feature importance or other attribute issue: {attr_err}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
