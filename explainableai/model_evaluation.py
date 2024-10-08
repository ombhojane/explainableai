# explainable_ai/model_evaluation.py

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix, classification_report
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

def train_keras_model(X, y):
    model = create_keras_model(X.shape[1])
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    return model

def evaluate_model(model, X, y, is_classifier):
    if is_classifier:
        return evaluate_classifier(model, X, y)
    else:
        return evaluate_regressor(model, X, y)

def evaluate_classifier(model, X, y):
    if hasattr(model, "predict_proba"):
        y_pred = model.predict(X)
    else:
        y_pred = (model.predict(X) > 0.5).astype(int)  # For Keras model, convert probabilities to binary predictions

    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y, y_pred)
    class_report = classification_report(y, y_pred)
    
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report
    }

def evaluate_regressor(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    return {
        "mean_squared_error": mse,
        "r2_score": r2
    }

def cross_validate(model, X, y, cv=5):
    if hasattr(model, "predict_proba"):
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    else:
        # For Keras model, we need to use a wrapper for cross-validation
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=cv)
        scores = []
        for train_index, test_index in kf.split(X):
            model.fit(X[train_index], y[train_index], epochs=10, batch_size=32, verbose=0)
            y_pred = (model.predict(X[test_index]) > 0.5).astype(int)
            score = accuracy_score(y[test_index], y_pred)
            scores.append(score)
        return np.mean(scores), np.std(scores)

# Example usage:
# X_train = ...  # your training feature data as a numpy array
# y_train = ...  # your training labels as a numpy array

# keras_model = train_keras_model(X_train, y_train)  # Train the Keras model
# evaluation_results = evaluate_model(keras_model, X_train, y_train, is_classifier=True)  # Evaluate the model