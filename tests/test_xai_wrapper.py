import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from explainableai import XAIWrapper

@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)
    feature_names = [f'feature_{i}' for i in range(20)]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name='target')
    return X, y

def test_xai_wrapper_initialization():
    xai = XAIWrapper()
    assert xai is not None

def test_xai_wrapper_fit(sample_data):
    X, y = sample_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    xai = XAIWrapper()
    xai.fit(model, X, y)
    assert xai.model is not None
    assert xai.X is not None
    assert xai.y is not None

def test_xai_wrapper_analyze(sample_data):
    X, y = sample_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    xai = XAIWrapper()
    xai.fit(model, X, y)
    results = xai.analyze()
    assert 'model_performance' in results
    assert 'feature_importance' in results
    assert 'llm_explanation' in results

def test_xai_wrapper_predict(sample_data):
    X, y = sample_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    xai = XAIWrapper()
    xai.fit(model, X, y)
    
    # Test single prediction
    single_input = X.iloc[0].to_dict()
    prediction, probabilities, explanation = xai.explain_prediction(single_input)
    assert isinstance(prediction, (int, np.integer))
    assert isinstance(probabilities, np.ndarray)
    assert isinstance(explanation, str)