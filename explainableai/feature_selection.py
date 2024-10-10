from sklearn.feature_selection import SelectKBest, f_classif
import logging

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def select_features(X, y, k=10):
    logger.debug("Selection the data...")
    try:
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_feature_indices = selector.get_support(indices=True)
        logger.info("Feature selected...")
        return X_selected, selected_feature_indices
    except Exception as e:
        logger.error(f"Some error occurred in feature selection...{str(e)}")