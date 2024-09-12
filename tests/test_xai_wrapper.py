import pytest
from explainableai import XAIWrapper
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. create a fixture that loads the sample data
# 2. test the initialization of the XAIWrapper class
# 3. test the fit method of the XAIWrapper class
# 4. test the analyze method of the XAIWrapper class

@pytest.fixture
def sample_data():
    df = pd.read_csv('datasets/cancer.csv')
    X = df.drop('Cancer', axis=1)
    y = df['Cancer']
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
