# fairness.py
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

def create_keras_model(input_dim):
    # Create a Keras model for binary classification
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_keras_model(X, y):
    model = create_keras_model(X.shape[1])
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    return model

def predict_with_keras(model, X):
    predictions = model.predict(X)
    return (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions

def demographic_parity(y_true, y_pred, protected_attribute):
    groups = np.unique(protected_attribute)
    group_probs = {}
    for group in groups:
        group_mask = protected_attribute == group
        group_probs[group] = np.mean(y_pred[group_mask])
    
    max_diff = max(group_probs.values()) - min(group_probs.values())
    return max_diff, group_probs

def equal_opportunity(y_true, y_pred, protected_attribute):
    groups = np.unique(protected_attribute)
    group_tpr = {}
    for group in groups:
        group_mask = protected_attribute == group
        tn, fp, fn, tp = confusion_matrix(y_true[group_mask], y_pred[group_mask]).ravel()
        group_tpr[group] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    max_diff = max(group_tpr.values()) - min(group_tpr.values())
    return max_diff, group_tpr

def disparate_impact(y_true, y_pred, protected_attribute):
    groups = np.unique(protected_attribute)
    group_probs = {}
    for group in groups:
        group_mask = protected_attribute == group
        group_probs[group] = np.mean(y_pred[group_mask])
    
    di = min(group_probs.values()) / max(group_probs.values())
    return di, group_probs

def plot_fairness_metrics(fairness_metrics):
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

# Example usage:
# X = ...  # your feature data as a numpy array
# y = ...  # your labels as a numpy array
# protected_attribute = ...  # your protected attribute as a numpy array

# model = train_keras_model(X, y)  # Train the Keras model
# y_pred = predict_with_keras(model, X)  # Make predictions
# dp, dp_values = demographic_parity(y, y_pred, protected_attribute)
# eo, eo_values = equal_opportunity(y, y_pred, protected_attribute)
# di, di_values = disparate_impact(y, y_pred, protected_attribute)

# fairness_metrics = {
#     'Demographic Parity': dp_values,
#     'Equal Opportunity': eo_values,
#     'Disparate Impact': di_values
# }
# plot_fairness_metrics(fairness_metrics)