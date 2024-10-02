from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error

def train_model(model_type, X, y, test_size=0.2, random_state=42, is_classifier=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if model_type == 'random_forest':
        model = RandomForestClassifier(random_state=random_state) if is_classifier else RandomForestRegressor(random_state=random_state)
    elif model_type == 'logistic_regression' and is_classifier:
        model = LogisticRegression(random_state=random_state)
    elif model_type == 'svm' and is_classifier:
        model = SVC(random_state=random_state)
    elif model_type == 'linear_regression' and not is_classifier:
        model = LinearRegression()
    else:
        raise ValueError("Unsupported model type or classifier/regressor mismatch.")

    model.fit(X_train, y_train)

    return model, X_test, y_test

def evaluate_model(model, X_test, y_test, is_classifier):
    y_pred = model.predict(X_test)

    if is_classifier:
        accuracy = accuracy_score(y_test, y_pred)
        return {"accuracy": accuracy}
    else:
        mse = mean_squared_error(y_test, y_pred)
        return {"mean_squared_error": mse}

def experiment_with_hyperparameters(model_type, X, y, param_grid, test_size=0.2, random_state=42, is_classifier=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if model_type == 'random_forest':
        model = RandomForestClassifier(random_state=random_state) if is_classifier else RandomForestRegressor(random_state=random_state)
    elif model_type == 'logistic_regression' and is_classifier:
        model = LogisticRegression(random_state=random_state)
    elif model_type == 'svm' and is_classifier:
        model = SVC(random_state=random_state)
    elif model_type == 'linear_regression' and not is_classifier:
        model = LinearRegression()
    else:
        raise ValueError("Unsupported model type or classifier/regressor mismatch.")

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy' if is_classifier else 'neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_
