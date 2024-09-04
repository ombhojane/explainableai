# main.py

import pandas as pd
from explainableai import XAIWrapper
import argparse

def main(file_path, target_column):
    # Import the dataset
    print("Importing dataset...")
    df = pd.read_csv(file_path)

    # Create XAIWrapper instance
    xai = XAIWrapper()

    # Run XAI analysis
    xai.analyze(df, target_column)

    # Test usage with user input
    while True:
        print("\nEnter values for prediction (or 'q' to quit):")
        user_input = {}
        for feature in xai.feature_names:
            value = input(f"{feature}: ")
            if value.lower() == 'q':
                return
            try:
                user_input[feature] = float(value)
            except ValueError:
                print(f"Invalid input for {feature}. Please enter a numeric value.")
                break
        else:
            try:
                prediction, probability = xai.predict(user_input)
                print("\nPrediction Results:")
                print(f"Prediction: {prediction}")
                print(f"Probability of positive class: {probability:.2f}")
            except Exception as e:
                print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XAI analysis on a dataset")
    parser.add_argument("file_path", help="Path to the CSV file containing the dataset")
    parser.add_argument("target_column", help="Name of the target column in the dataset")
    args = parser.parse_args()

    main(args.file_path, args.target_column)