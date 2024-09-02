# main.py

from explainable_ai import xai_wrapper, analyze_dataset, preprocess_data
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import argparse

@xai_wrapper
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# main.py

def main(file_path, target_column):
    # Import and analyze the dataset
    print("Importing and analyzing dataset...")
    df = pd.read_csv(file_path)
    analyze_dataset(df)

    # Preprocess the data
    print("Preprocessing data...")
    X, y = preprocess_data(df, target_column)

    # Train the model with XAI wrapper
    print("Training model with XAI wrapper...")
    model, xai_results = train_model(X, y)

    print("Model:", model)
    print("\nModel Metrics:", xai_results['metrics'])
    print("\nTop 5 Important Features:")
    top_features = sorted(xai_results['explanation']['feature_importance'].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    for feature, importance in top_features:
        print(f"{feature}: {importance:.4f}")

    print(f"\nCross-validation Score: {xai_results['cv_score']:.4f} (+/- {xai_results['cv_std']:.4f})")

    print("\nModel Comparison:")
    for model_name, scores in xai_results['model_comparison'].items():
        print(f"{model_name}: CV Score = {scores['cv_score']:.4f}, Test Score = {scores['test_score']:.4f}")

    print(f"\nAnomalies detected: {sum(xai_results['anomalies'] == -1)}")

    if xai_results['shap_values'] is not None:
        print("\nSHAP values calculated successfully. See 'shap_summary.png' for visualization.")
    else:
        print("\nSHAP values calculation failed. Please check the console output for more details.")

    print("\nVisualizations saved:")
    print("- Feature Importance: feature_importance.png")
    print("- Partial Dependence: partial_dependence.png")
    print("- Learning Curve: learning_curve.png")
    print("- ROC Curve: roc_curve.png")
    print("- Precision-Recall Curve: precision_recall_curve.png")
    print("- Calibration Curve: calibration_curve.png")
    print("- Model Comparison ROC Curves: model_comparison_roc_curves.png")

    print("\nAnalyzing feature interactions...")
    for i, (f1, f2, _) in enumerate(xai_results['interactions']):
        print(f"Interaction {i+1}: {f1} and {f2}")

    # Example of how to use the model for prediction
    print("\nPreparing example prediction...")
    def predict(model, patient_data):
        prediction = model.predict(patient_data)[0]
        probabilities = model.predict_proba(patient_data)[0]
        return prediction, probabilities

    # Example data (ensure it's preprocessed the same way as the training data)
    new_data = xai_results['X_test_selected'][0].reshape(1, -1)  # Using the first test sample as an example
    prediction, probabilities = predict(model, new_data)

    print("\nNew Data Prediction:")
    print(f"Prediction: {prediction}")
    print(f"Probability of positive class: {probabilities[1]:.2f}")

    print("\nXAI analysis completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XAI analysis on a dataset")
    parser.add_argument("file_path", help="Path to the CSV file containing the dataset")
    parser.add_argument("target_column", help="Name of the target column in the dataset")
    args = parser.parse_args()

    main(args.file_path, args.target_column)