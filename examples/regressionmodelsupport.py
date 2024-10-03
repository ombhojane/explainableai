import pandas as pd
from explainableai import XAIWrapper
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

# Load your dataset
df = pd.read_csv('your_dataset.csv')
X = df.drop(columns=['target_column'])
y = df['target_column']

# Perform EDA
XAIWrapper.perform_eda(df)

# Create models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression(),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

# Create XAIWrapper instance and fit models
xai = XAIWrapper()
xai.fit(models, X, y)
results = xai.analyze()

# Print LLM explanation of results
print(results['llm_explanation'])

# Generate a comprehensive report
xai.generate_report('xai_report.pdf')

# Make a prediction with explanation
new_data = {...}  # Dictionary of feature values
prediction, probabilities, explanation = xai.explain_prediction(new_data)
print(f"Prediction: {prediction}")
print(f"Probabilities: {probabilities}")
print(f"Explanation: {explanation}")