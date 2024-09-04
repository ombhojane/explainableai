from sklearn.ensemble import IsolationForest

def detect_anomalies(X):
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = iso_forest.fit_predict(X)
    return anomalies