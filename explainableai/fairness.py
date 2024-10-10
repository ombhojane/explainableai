# fairness.py
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import logging

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def demographic_parity(y_true, y_pred, protected_attribute):
    logger.debug("Calculating demographic praity...")
    try:
        groups = np.unique(protected_attribute)
        group_probs = {}
        for group in groups:
            group_mask = protected_attribute == group
            group_probs[group] = np.mean(y_pred[group_mask])
    
        max_diff = max(group_probs.values()) - min(group_probs.values())
        return max_diff, group_probs
    except Exception as e:
        logger.error(f"Error occurred by calculating demographic paraity...{str(e)}")

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
        logger.error(f"Error occurred while calulating equal opportunity...{str(e)}")

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
        logger.error(f"Error occurred while Calculating disparate impact...{str(e)}")
def plot_fairness_metrics(fairness_metrics):
    logger.debug("Plotting fairness metrics...")
    try:
        plt.figure(figsize=(12, 6))
        for metric, values in fairness_metrics.items():
            plt.bar(range(len(values)), list(values.values()), label=metric)
            plt.xticks(range(len(values)), list(values.keys()))
    
        plt.xlabel('Protected Group')
        plt.ylabel('Metric Value')
        plt.title('Fairness Metrics Across Protected Groups')
        plt.legend()
        plt.tight_layout()
        plt.savefig('fairness_metrics.png')
        plt.close()
    except Exception as e:
        logger.error(f"Error occurred while Plotting fairness metrics...{str(e)}")