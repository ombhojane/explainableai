import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from explainableai.utils import explain_model, calculate_metrics
from dotenv import load_dotenv
import os 
load_dotenv()

def test_explain_model_regression():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    explanation = explain_model(model, X_train, y_train, X_test, y_test, feature_names)
    
    assert "feature_importance" in explanation
    assert "model_type" in explanation
    assert explanation["model_type"] == str(type(model))
    assert len(explanation["feature_importance"]) == X.shape[1]

def test_explain_model_classification():
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    explanation = explain_model(model, X_train, y_train, X_test, y_test, feature_names)
    
    assert "feature_importance" in explanation
    assert "model_type" in explanation
    assert explanation["model_type"] == str(type(model))
    assert len(explanation["feature_importance"]) == X.shape[1]

def test_calculate_metrics_regression():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    metrics = calculate_metrics(model, X_test, y_test)
    
    assert "mse" in metrics
    assert "r2" in metrics

def test_calculate_metrics_classification():
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    metrics = calculate_metrics(model, X_test, y_test)
    
    assert "accuracy" in metrics
    assert "f1_score" in metrics

if __name__ == "__main__":
    pytest.main()