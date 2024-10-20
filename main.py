import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from explainableai import XAIWrapper
import argparse
import logging

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(file_path):
    """Load dataset from a CSV file."""
    try:
        logger.info("Loading dataset...")
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def split_data(df, target_column):
    """Split dataset into training and testing sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_models():
    """Create and return a dictionary of ML models."""
    logger.info("Initializing models...")
    return {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }

def generate_predictions(xai, X):
    """Prompt user for input and generate predictions."""
    while True:
        logger.info("\nEnter values for prediction (or 'q' to quit):")
        user_input = {}
        for feature in X.columns:
            value = input(f"{feature}: ")
            if value.lower() == 'q':
                return
            try:
                user_input[feature] = float(value)
            except ValueError:
                user_input[feature] = value
        
        try:
            prediction, probabilities, explanation = xai.explain_prediction(user_input)
            logger.info("\nPrediction Results:")
            logger.info(f"Prediction: {prediction}")
            logger.info(f"Probabilities: {probabilities}")
            logger.info("\nExplanation of Prediction:")
            logger.info(explanation)
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")

def main(file_path, target_column):
    # Load and explore the dataset
    df = load_dataset(file_path)
    XAIWrapper.perform_eda(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df, target_column)
    
    # Create models
    models = create_models()

    # XAI wrapper initialization and model fitting
    xai = XAIWrapper()
    logger.info("Fitting models and performing XAI analysis...")
    xai.fit(models, X_train, y_train)
    results = xai.analyze()

    logger.info("\nModel Analysis Results:")
    logger.info(results['llm_explanation'])

    # Generate report
    try:
        logger.info("Generating XAI report...")
        xai.generate_report()
    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")

    # Prediction based on user input
    generate_predictions(xai, X_train)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XAI analysis on a dataset")
    parser.add_argument("file_path", help="Path to the CSV file containing the dataset")
    parser.add_argument("target_column", help="Name of the target column in the dataset")
    args = parser.parse_args()

    logger.info("Starting the XAI program...")
    main(args.file_path, args.target_column)