from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
def detect_anomalies(X):
    try:
        #Creating an Isolation forest
        logger.debug("Creating isolation forest model...")
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        
        #Prediction
        logger.debug("Making Prediction...")
        anomalies = iso_forest.fit_predict(X)
        logger.info("Prediction Maked...")
        return anomalies
    except Exception as e:
        logger.error(f"Something wrong in the prediction...{str(e)}")
