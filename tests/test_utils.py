import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from explainableai.utils import explain_model, calculate_metrics
from dotenv import load_dotenv
import os 

# Load environment variables
load_dotenv()

# Helper function to split data
def get_train_test_split(data_func, test_size=0.2, random_state=42, **kwargs):
    X, y = data_func(**kwargs)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Helper function for generating feature names
def generate_feature_names(X):
    return [f"feature_{i}" for i in range(X.shape[1])]

# Test for regression model explanation
def test_explain_model_regression():
    """Test if explain_model function works correctly for regression."""
    X_train, X_test, y_train, y_test = get_train_test_split(make_regression, 
                                                            n_samples=100, 
                                                            n_features=5, 
                                                            noise=0.1)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    feature_names = generate_feature_names(X_train)
    
    explanation = explain_model(model, X_train, y_train, X_test, y_test, feature_names)
    
    assert "feature_importance" in explanation, "Feature importance not found in explanation."
    assert "model_type" in explanation, "Model type not found in explanation."
    assert explanation["model_type"] == str(type(model)), f"Expected model type {type(model)}, but got {explanation['model_type']}"
    assert len(explanation["feature_importance"]) == X_train.shape[1], "Incorrect feature importance length."

# Test for classification model explanation
def test_explain_model_classification():
    """Test if explain_model function works correctly for classification."""
    X_train, X_test, y_train, y_test = get_train_test_split(make_classification, 
                                                            n_samples=100, 
                                                            n_features=5, 
                                                            n_classes=2)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    feature_names = generate_feature_names(X_train)
    
    explanation = explain_model(model, X_train, y_train, X_test, y_test, feature_names)
    
    assert "feature_importance" in explanation, "Feature importance not found in explanation."
    assert "model_type" in explanation, "Model type not found in explanation."
    assert explanation["model_type"] == str(type(model)), f"Expected model type {type(model)}, but got {explanation['model_type']}"
    assert len(explanation["feature_importance"]) == X_train.shape[1], "Incorrect feature importance length."

# Test for regression model metrics calculation
def test_calculate_metrics_regression():
    """Test if calculate_metrics function works correctly for regression models."""
    X_train, X_test, y_train, y_test = get_train_test_split(make_regression, 
                                                            n_samples=100, 
                                                            n_features=5, 
                                                            noise=0.1)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    metrics = calculate_metrics(model, X_test, y_test)
    
    assert "mse" in metrics, "MSE not found in metrics."
    assert "r2" in metrics, "R-squared not found in metrics."

# Test for classification model metrics calculation
def test_calculate_metrics_classification():
    """Test if calculate_metrics function works correctly for classification models."""
    X_train, X_test, y_train, y_test = get_train_test_split(make_classification, 
                                                            n_samples=100, 
                                                            n_features=5, 
                                                            n_classes=2)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    metrics = calculate_metrics(model, X_test, y_test)
    
    assert "accuracy" in metrics, "Accuracy not found in metrics."
    assert "f1_score" in metrics, "F1 score not found in metrics."