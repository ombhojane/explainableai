from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
import logging

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def automated_feature_engineering(X_train, X_test=None):
    # Convert categorical variables to one-hot encoding
    logger.debug("Convert categorical variables to one-hot encoding...")
    try:
        X_train_encoded = pd.get_dummies(X_train, drop_first=True)
        
        if X_test is not None:
            X_test_encoded = pd.get_dummies(X_test, drop_first=True)
            # Ensure X_test has the same columns as X_train
            logger.debug("Ensuring test data has the same columns as training data...")
            for col in X_train_encoded.columns:
                if col not in X_test_encoded.columns:
                    X_test_encoded[col] = 0
            X_test_encoded = X_test_encoded[X_train_encoded.columns]
            
        
        feature_names = X_train_encoded.columns.tolist()
        
        if X_test is not None:
            logger.info("Data Converted...")
            return X_train_encoded.values, X_test_encoded.values, feature_names
        
        logger.info("Data Converted...")
        return X_train_encoded.values, feature_names
    except Exception as e:
        logger.error(f"Error occurred during automated feature engineering...{str(e)}")