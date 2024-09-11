import pytest
from explainableai import XAIWrapper
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture
def sample_data():
    # Create a simple dataset for testing
    df = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target': np.random.randint(0, 2, 100)
    })
    X = df.drop('target', axis=1)
    y = df['target']
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