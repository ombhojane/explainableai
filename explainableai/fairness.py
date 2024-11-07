import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def demographic_parity(y_true, y_pred, protected_attribute):
    logger.debug("Calculating demographic parity...")
    try:
        groups = np.unique(protected_attribute)
        group_probs = {}
        for group in groups:
            group_mask = protected_attribute == group
            group_probs[group] = np.mean(y_pred[group_mask])
    
        max_diff = max(group_probs.values()) - min(group_probs.values())
        return max_diff, group_probs
    except Exception as e:
        logger.error(f"Error occurred by calculating demographic parity...{str(e)}")

def equal_opportunity(y_true, y_pred, protected_attribute):
    logger.debug("Checking for equal opportunity...")
    try:
        groups = np.unique(protected_attribute)
        group_tpr = {}
        for group in groups:
            group_mask = protected_attribute == group
            tn, fp, fn, tp = confusion_matrix(y_true[group_mask], y_pred[group_mask]).ravel()
            group_tpr[group] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
        max_diff = max(group_tpr.values()) - min(group_tpr.values())
        return max_diff, group_tpr
    except Exception as e:
        logger.error(f"Error occurred while calculating equal opportunity...{str(e)}")

def disparate_impact(y_true, y_pred, protected_attribute):
    logger.debug("Calculating disparate impact...")
    try:
        groups = np.unique(protected_attribute)
        group_probs = {}
        for group in groups:
            group_mask = protected_attribute == group
            group_probs[group] = np.mean(y_pred[group_mask])
    
        di = min(group_probs.values()) / max(group_probs.values())
        return di, group_probs
    except Exception as e:
        logger.error(f"Error occurred while calculating disparate impact...{str(e)}")

def plot_fairness_metrics(fairness_metrics, colors=None):
    logger.debug("Plotting fairness metrics...")
    try:
        plt.figure(figsize=(12, 6))
        
        # If no specific colors are provided, use a default color palette
        if colors is None:
            colors = plt.cm.get_cmap('tab10', len(fairness_metrics))

        for idx, (metric, values) in enumerate(fairness_metrics.items()):
            color = colors(idx)  # Get color from the palette
            
            # Plot the bars for each metric
            plt.bar(range(len(values)), list(values.values()), label=metric, color=color)
            plt.xticks(range(len(values)), list(values.keys()))

            # Dynamically add metric name to title
            plt.title(f'Fairness Metric: {metric}', fontsize=14)
    
        plt.xlabel('Protected Group', fontsize=12)
        plt.ylabel('Metric Value', fontsize=12)
        plt.legend(title="Fairness Metrics", loc='upper left', fontsize=10)
        plt.tight_layout()

        # Save the plot with a descriptive filename
        plt.savefig('fairness_metrics.png')
        plt.close()
        logger.info("Fairness metrics plot saved successfully.")
    
    except Exception as e:
        logger.error(f"Error occurred while plotting fairness metrics...{str(e)}")
