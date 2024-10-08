# model_comparison.py
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
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

def compare_models(models, X, y, cv=5):
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        results[name] = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
        }
    return results

def plot_roc_curves(models, X, y):
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X)[:, 1]
        else:
            # For Keras model, use the predict method
            y_pred_proba = model.predict(X).flatten()  # Flatten to get 1D array
        
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        auc = roc_auc_score(y, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Model Comparison')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves_comparison.png')
    plt.close()

def mcnemar_test(model1, model2, X, y):
    y_pred1 = model1.predict(X)
    y_pred2 = model2.predict(X)
    
    # Convert probabilities to binary predictions for Keras models
    if y_pred1.ndim > 1:
        y_pred1 = (y_pred1 > 0.5).astype(int).flatten()
    if y_pred2.ndim > 1:
        y_pred2 = (y_pred2 > 0.5).astype(int).flatten()
    
    contingency_table = np.zeros((2, 2))
    contingency_table[0, 0] = np.sum((y_pred1 == y) & (y_pred2 == y))
    contingency_table[0, 1] = np.sum((y_pred1 == y) & (y_pred2 != y))
    contingency_table[1, 0] = np.sum((y_pred1 != y) & (y_pred2 == y))
    contingency_table[1, 1] = np.sum((y_pred1 != y) & (y_pred2 != y))
    
    statistic, p_value = stats.mcnemar(contingency_table, correction=True)
    return statistic, p_value

# Example usage:
# X_train = ...  # your training feature data as a numpy array
# y_train = ...  # your training labels as a numpy array

# keras_model = train_keras_model(X_train, y_train)  # Train the Keras model
# models = {'Keras Model': keras_model, 'Other Model': other_model}  # Add other models as needed
# results = compare_models(models, X_train, y_train)  # Compare models
# plot_roc_curves(models, X_train, y_train)  # Plot ROC curves