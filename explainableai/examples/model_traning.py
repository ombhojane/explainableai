from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error

def train_model(model_type, X, y, test_size=0.2, random_state=42, is_classifier=True):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize the model based on the specified type
    if model_type == 'random_forest':
        # Use RandomForestClassifier for classification or RandomForestRegressor for regression
        model = RandomForestClassifier(random_state=random_state) if is_classifier else RandomForestRegressor(random_state=random_state)
    elif model_type == 'logistic_regression' and is_classifier:
        model = LogisticRegression(random_state=random_state)
    elif model_type == 'svm' and is_classifier:
        model = SVC(random_state=random_state)
    elif model_type == 'linear_regression' and not is_classifier:
        model = LinearRegression()
    else:
        # Raise an error for unsupported model types or classifier/regressor mismatch
        raise ValueError("Unsupported model type or classifier/regressor mismatch.")

    # Train the model on the training set
    model.fit(X_train, y_train)

    # Return the trained model along with the test set
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test, is_classifier):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate model performance based on whether it's a classifier or regressor
    if is_classifier:
        accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy for classifiers
        return {"accuracy": accuracy}
    else:
        mse = mean_squared_error(y_test, y_pred)  # Calculate mean squared error for regressors
        return {"mean_squared_error": mse}

def experiment_with_hyperparameters(model_type, X, y, param_grid, test_size=0.2, random_state=42, is_classifier=True):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize the model based on the specified type
    if model_type == 'random_forest':
        model = RandomForestClassifier(random_state=random_state) if is_classifier else RandomForestRegressor(random_state=random_state)
    elif model_type == 'logistic_regression' and is_classifier:
        model = LogisticRegression(random_state=random_state)
    elif model_type == 'svm' and is_classifier:
        model = SVC(random_state=random_state)
    elif model_type == 'linear_regression' and not is_classifier:
        model = LinearRegression()
    else:
        # Raise an error for unsupported model types or classifier/regressor mismatch
        raise ValueError("Unsupported model type or classifier/regressor mismatch.")

    # Set up the GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy' if is_classifier else 'neg_mean_squared_error')
    
    # Fit the grid search on the training data
    grid_search.fit(X_train, y_train)

    # Return the best estimator found during the grid search
    return grid_search.best_estimator_
