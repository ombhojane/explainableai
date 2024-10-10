# SNP 500 USING explainableai

import pandas as pd
from explainableai import XAIWrapper
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the mock data
df = pd.read_csv('mock_snp500_data.csv', index_col=0)

# Prepare features and target
X = df.drop(columns=['S&P500'])  # Features (Open, High, Low, Close, Volume)
y = df['S&P500']  # Target (S&P500 index)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the ExplainableAI wrapper
xai = XAIWrapper()

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model and analyze
xai.fit(model, X_train, y_train)
results = xai.analyze(X_test, y_test)

# Print LLM-powered explanation of the results
print("LLM Explanation of the Results:")
print(results['llm_explanation'])

# Generate a detailed PDF report
xai.generate_report('snp500_analysis_report.pdf')

# Make an individual prediction with explanation (using a sample from the test set)
new_data = X_test.iloc[0].to_dict()
prediction, probabilities, explanation = xai.explain_prediction(new_data)

print(f"Prediction for individual sample: {prediction}")
print(f"Explanation for prediction: {explanation}")

# Indicate where the report is stored
print("Report generated as 'snp500_analysis_report.pdf'")
