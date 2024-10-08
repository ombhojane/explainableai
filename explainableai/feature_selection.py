from sklearn.feature_selection import SelectKBest, f_classif
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

def select_features(X, y, k=10):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_feature_indices = selector.get_support(indices=True)
    return X_selected, selected_feature_indices

# Example usage:
# X = ...  # your feature data as a numpy array
# y = ...  # your labels as a numpy array

# X_selected, selected_indices = select_features(X, y, k=10)  # Select features
# model = train_keras_model(X_selected, y)  # Train the Keras model
# predictions = predict_with_keras(model, X_selected)  # Make predictions