# explainable_ai/model_evaluation.py

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
import logging

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def evaluate_model(model, X, y, is_classifier):
    logger.debug("Evaluting model")
    try:
        if is_classifier:
            return evaluate_classifier(model, X, y)
        else:
            return evaluate_regressor(model, X, y)
    except Exception as e:
        logger.error(f"Some error occurred in evaluting model...{str(e)}")

def evaluate_classifier(model, X, y):
    logger.debug("Evaluating report...")
    try:
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y, y_pred)
        class_report = classification_report(y, y_pred)
        logger.info("Report Generated...")
        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report
        }
    except Exception as e:
        logger.error(f"Some error occured in evaluating report...{str(e)}")
def evaluate_regressor(model, X, y):
    logger.debug("Model prediction...")
    try:
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        logger.info("Model predicted...")
        return {
            "mean_squared_error": mse,
            "r2_score": r2
        }
    except Exception as e:
        logger.error(f"Some error in model prediction...{str(e)}")

def cross_validate(model, X, y, cv=5):
    logger.debug("Cross validation...")
    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        logger.info("validated...")
        return scores.mean(), scores.std()
    except Exception as e:
        logger.error(f"Some error in validation...{str(e)}")