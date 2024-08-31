# main.py
from explainable_ai import xai_wrapper, analyze_dataset
from explainable_ai.core import preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

@xai_wrapper
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Import and analyze the dataset
df = analyze_dataset('heart_disease_dataset.csv')

# Preprocess the data
X, y = preprocess_data(df, target_column='heart_disease')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model with XAI wrapper
# Train the model with XAI wrapper
model, xai_results = train_model(X_train, y_train, X_test=X_test, y_test=y_test)

print("Model:", model)
print("\nModel Metrics:", xai_results['metrics'])
print("\nTop 5 Important Features:")
top_features = sorted(xai_results['explanation']['feature_importance'].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
for feature, importance in top_features:
    print(f"{feature}: {importance:.4f}")

print("\nModel Comparison:")
for model_name, scores in xai_results['model_comparison'].items():
    print(f"{model_name}: CV Score = {scores['cv_score']:.4f}, Test Score = {scores['test_score']:.4f}")

print(f"\nAnomalies detected: {sum(xai_results['anomalies'] == -1)}")

if xai_results['shap_values'] is not None:
    print("\nSHAP values calculated successfully. See 'shap_summary.png' for visualization.")
else:
    print("\nSHAP values calculation failed. Please check the data and model compatibility.")

# Example of how to use the model for prediction
def predict(model, patient_data):
    prediction = model.predict(patient_data)[0]
    probabilities = model.predict_proba(patient_data)[0]
    return prediction, probabilities

# Example data (ensure it's preprocessed the same way as the training data)
new_data = xai_results['X_test_fe'][7].reshape(1, -1) 
prediction, probabilities = predict(model, new_data)

print("\nNew Data Prediction:")
print(f"Prediction: {prediction}")
print(f"Probability of heart disease: {probabilities[1]:.2f}")