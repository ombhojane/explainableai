# model_selection.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
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

def get_default_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
        'Keras Model': None  # Placeholder for Keras model
    }

def compare_models(X_train, y_train, X_test, y_test, models=None):
    if models is None:
        models = get_default_models()
    
    results = {}
    for name, model in models.items():
        if name == 'Keras Model':
            keras_model = create_keras_model(X_train.shape[1])
            keras_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
            y_pred_proba = keras_model.predict(X_test).flatten()  # Flatten to get 1D array
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            results[name] = {
                'cv_score': None,  # Keras model does not use cross_val_score in this context
                'test_score': roc_auc
            }
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            model.fit(X_train, y_train)
            test_score = model.score(X_test, y_test)
            results[name] = {
                'cv_score': cv_scores.mean(),
                'test_score': test_score
            }
    
    plot_roc_curves(models, X_test, y_test)
    
    return results

def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        if name == 'Keras Model':
            y_pred_proba = model.predict(X_test).flatten()  # Use the Keras model's predictions
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('model_comparison_roc_curves.png')
    plt.close()

# Example usage:
# X_train = ...  # your training feature data as a numpy array
# y_train = ...  # your training labels as a numpy array
# X_test = ...   # your testing feature data as a numpy array
# y_test = ...   # your testing labels as a numpy array

# results = compare_models(X_train, y_train, X_test, y_test)  # Compare models