# main.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from explainableai import XAIWrapper
import argparse

def main(file_path, target_column):
    # Import the dataset
    print("Importing dataset...")
    df = pd.read_csv(file_path)
    
    # Perform EDA
    XAIWrapper.perform_eda(df)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }

    # Create XAIWrapper instance
    xai = XAIWrapper()

    # Fit the models and run XAI analysis
    xai.fit(models, X_train, y_train)
    results = xai.analyze()

    print("\nLLM Explanation of Results:")
    print(results['llm_explanation'])

    # Generate the report
    try:
        xai.generate_report()
    except Exception as e:
        print(f"An error occurred while generating the report: {str(e)}")

    # Example of using the trained model for new predictions
    while True:
        print("\nEnter values for prediction (or 'q' to quit):")
        user_input = {}
        for feature in X.columns:
            value = input(f"{feature}: ")
            if value.lower() == 'q':
                return
            # Try to convert to float if possible, otherwise keep as string
            try:
                user_input[feature] = float(value)
            except ValueError:
                user_input[feature] = value
        
        try:
            prediction, probabilities, explanation = xai.explain_prediction(user_input)
            print("\nPrediction Results:")
            print(f"Prediction: {prediction}")
            print(f"Probabilities: {probabilities}")
            print("\nLLM Explanation of Prediction:")
            print(explanation)
        except Exception as e:
            print(f"An error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XAI analysis on a dataset")
    parser.add_argument("file_path", help="Path to the CSV file containing the dataset")
    parser.add_argument("target_column", help="Name of the target column in the dataset")
    args = parser.parse_args()

    main(args.file_path, args.target_column)