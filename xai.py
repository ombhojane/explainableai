from explainable_ai import xai_wrapper
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

@xai_wrapper
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Generate sample data
np.random.seed(42)
X = np.random.rand(1000, 5)
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + 0.5*X[:, 3] - 1.5*X[:, 4] + np.random.normal(0, 0.1, 1000)

# Convert to DataFrame for better feature naming
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model with XAI wrapper
model, xai_results = train_model(X_train, y_train, X_test=X_test, y_test=y_test)

print("Model:", model)
print("Explanation:", xai_results['explanation'])
print("Metrics:", xai_results['metrics'])
print("SHAP Values Shape:", xai_results['shap_values'].shape)