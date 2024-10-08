from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.inspection import permutation_importance
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

def train_keras_model(X_train, y_train):
    model = create_keras_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    return model

def explain_model(model, X_train, y_train, X_test, y_test, feature_names):
    if isinstance(model, keras.Model):  # Check if the model is a Keras model
        y_pred = (model.predict(X_test) > 0.5).astype(int)  # Convert probabilities to binary predictions
    else:
        y_pred = model.predict(X_test)

    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    feature_importance = {feature: importance for feature, importance in zip(feature_names, result.importances_mean)}
    
    # Sort feature importance by absolute value
    feature_importance = dict(sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True))
    
    return {
        "feature_importance": feature_importance,
        "model_type": str(type(model)),
    }

def calculate_metrics(model, X_test, y_test):
    if isinstance(model, keras.Model):  # Check if the model is a Keras model
        y_pred = (model.predict(X_test) > 0.5).astype(int)  # Convert probabilities to binary predictions
    else:
        y_pred = model.predict(X_test)
    
    if len(np.unique(y_test)) == 2:  # Binary classification
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average='weighted')
        }
    else:  # Regression or multi-class classification
        return {
            "mse": mean_squared_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred)
        }

# Example usage:
# X_train = ...  # your training feature data as a numpy array
# y_train = ...  # your training labels as a numpy array
# X_test = ...   # your testing feature data as a numpy array
# y_test = ...   # your testing labels as a numpy array

# keras_model = train_keras_model(X_train, y_train)  # Train the Keras model
# metrics = calculate_metrics(keras_model, X_test, y_test)  # Calculate metrics
# feature_importance = explain_model(keras_model, X_train, y_train, X_test, y_test, feature_names)  # Explain model