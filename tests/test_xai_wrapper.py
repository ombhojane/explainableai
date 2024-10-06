import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from explainableai.core import XAIWrapper
import logging

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@pytest.fixture
def sample_data():
    logger.debug("Generating sample data...")
    try:
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)
        feature_names = [f'feature_{i}' for i in range(20)]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name='target')
        logger.info("Sample data generated.")
        return X, y
    except Exception as e:
        logger.error(f"Some error occurred while generating sample data: {str(e)}")

@pytest.fixture
def sample_models():
    logger.debug("Defining sample models...")
    try:
        return {
            'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'XGBoost': XGBClassifier(n_estimators=10, random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
        }
    except Exception as e:
        logger.error(f"Some error occured while defining sample models: {str(e)}")
def test_xai_wrapper_initialization(sample_data, sample_models):
    logger.debug("Testing XAIWrapper initialization...")
    try:
        X, y = sample_data
        xai = XAIWrapper()
        assert xai is not None
        assert xai.model is None
        assert xai.X is None
        assert xai.y is None
        logger.info("XAIWrapper initialization test passed.")
    except Exception as e:
        logger.error(f"An error occurred while testing XAIWrapper initialization: {str(e)}")

def test_xai_wrapper_fit(sample_data, sample_models):
    logger.debug("Testing XAIWrapper fit method...")
    try:
        X, y = sample_data
        xai = XAIWrapper()
        xai.fit(sample_models, X, y)
        assert xai.model is not None
        assert xai.X is not None
        assert xai.y is not None
        assert hasattr(xai.model, 'predict')
        assert hasattr(xai.model, 'predict_proba')
        logger.info("XAIWrapper fit method test passed.")
    except Exception as e:
        logger.error(f"An error occurred while testing XAIWrapper fit method: {str(e)}")

@pytest.mark.parametrize("model_name", ['Random Forest', 'Logistic Regression', 'XGBoost', 'Neural Network'])
def test_xai_wrapper_analyze_with_different_models(sample_data, sample_models, model_name):
    logger.debug(f"Testing XAIWrapper analyze method with {model_name}...")
    try:
        X, y = sample_data
        models = {model_name: sample_models[model_name]}
        xai = XAIWrapper()
        xai.fit(models, X, y)
        results = xai.analyze()
        assert 'model_performance' in results
        assert 'feature_importance' in results
        assert 'shap_values' in results
        assert 'cv_scores' in results
        assert 'llm_explanation' in results
        assert 'model_comparison' in results
        logger.info(f"XAIWrapper analyze method test with {model_name} passed.")
    except Exception as e:
        logger.error(f"An error occurred while testing XAIWrapper analyze method with {model_name}: {str(e)}")

def test_xai_wrapper_predict(sample_data, sample_models):
    logger.debug("Testing XAIWrapper predict method...")
    try:
        X, y = sample_data
        xai = XAIWrapper()
        xai.fit(sample_models, X, y)
        
        # Test single prediction
        single_input = X.iloc[0].to_dict()
        prediction, probabilities, explanation = xai.explain_prediction(single_input)
        assert isinstance(prediction, (int, np.integer))
        assert isinstance(probabilities, np.ndarray)
        assert isinstance(explanation, str)
        logger.info("XAIWrapper predict method test passed.")
    except Exception as e:
        logger.error(f"An error occurred while testing XAIWrapper predict method: {str(e)}")

def test_xai_wrapper_generate_report(sample_data, sample_models, tmp_path):
    logger.debug("Testing XAIWrapper generate report...")
    try:
        X, y = sample_data
        xai = XAIWrapper()
        xai.fit(sample_models, X, y)
        xai.analyze()
        
        report_path = tmp_path / "test_report.pdf"
        xai.generate_report(filename=str(report_path))
        assert report_path.exists()
        assert os.path.getsize(report_path) > 0  # Check if the file is not empty
        logger.info("XAIWrapper generate report test passed.")
    except Exception as e:
        logger.error(f"An error occurred while testing XAIWrapper generate report: {str(e)}")

def test_xai_wrapper_perform_eda(sample_data):
    logger.debug("Testing XAIWrapper perform eda...")
    try:
        X, y = sample_data
        df = pd.concat([X, y], axis=1)
        try:
            XAIWrapper.perform_eda(df)
        except Exception as e:
            pytest.fail(f"perform_eda raised an exception: {e}")
    except Exception as e:
        logger.error(f"An error occurred while testing XAIWrapper perform data: {str(e)}")

def test_xai_wrapper_feature_importance(sample_data, sample_models):
    logger.debug("Testing XAIWrapper feature importance...")
    try:
        X, y = sample_data
        xai = XAIWrapper()
        xai.fit(sample_models, X, y)
        results = xai.analyze()
        assert 'feature_importance' in results
        assert len(results['feature_importance']) == X.shape[1]
        assert all(isinstance(importance, (float, np.float64)) for importance in results['feature_importance'].values())
        logger.info("XAIWrapper feature importance test passed.")
    except Exception as e:
        logger.error(f"An error occurred while testing XAIWrapper feature importance: {str(e)}")

def test_xai_wrapper_cross_validation(sample_data, sample_models):
    logger.debug("Testing XAIWrapper cross validation...")
    try:
        X, y = sample_data
        xai = XAIWrapper()
        xai.fit(sample_models, X, y)
        results = xai.analyze()
        assert 'cv_scores' in results
        assert len(results['cv_scores']) == 2  # mean and std
        assert all(isinstance(score, (float, np.float64)) for score in results['cv_scores'])
        logger.info("XAIWrapper cross validation test passed.")
    except Exception as e:
        logger.error(f"An error occurred while testing XAIWrapper cross validation: {str(e)}")

def test_xai_wrapper_model_comparison(sample_data, sample_models):
    logger.debug("Testing XAIWrapper model comparision...")
    try:
        X, y = sample_data
        xai = XAIWrapper()
        xai.fit(sample_models, X, y)
        results = xai.analyze()
        assert 'model_comparison' in results
        assert len(results['model_comparison']) == len(sample_models)
        for model_name, scores in results['model_comparison'].items():
            assert 'cv_score' in scores
            assert 'test_score' in scores
            assert isinstance(scores['cv_score'], (float, np.float64))
            assert isinstance(scores['test_score'], (float, np.float64))
        logger.info("XAIWrapper model comparision test passed.")
    except Exception as e:
        logger.error(f"An error occurred while testing XAIWrapper model comparision: {str(e)}")

@pytest.mark.parametrize("invalid_input", [
    {},  # Empty dictionary
    {'invalid_feature': 1},  # Invalid feature name
    {f'feature_{i}': 'invalid' for i in range(20)},  # Invalid data type
])
def test_xai_wrapper_predict_invalid_input(sample_data, sample_models, invalid_input):
    logger.debug("Testing XAIWrapper predict invalid input...")
    try:
        X, y = sample_data
        xai = XAIWrapper()
        xai.fit(sample_models, X, y)
        with pytest.raises(Exception):
            xai.explain_prediction(invalid_input)
    except Exception as e:
        logger.error(f"An error occurred while testing XAIWrapper predict invalid input: {str(e)}")