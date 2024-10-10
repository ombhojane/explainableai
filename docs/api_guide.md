# ExplainableAI API Reference Guide

## XAIWrapper Class

The primary interface for the ExplainableAI package. This class wraps machine learning models and provides various explainability methods.

### Constructor

```python
xai = XAIWrapper()
```

### Core Methods

#### 1. fit()

```python
xai.fit(models, X, y, feature_names=None)
```

Fits the model(s) and prepares for analysis.

**Parameters:**
- `models`: Single model or dictionary of models
  - Type: sklearn-compatible model or dict of models
- `X`: Training data features
  - Type: numpy array or pandas DataFrame
- `y`: Target variable
  - Type: numpy array or pandas Series
- `feature_names`: Names of features (optional)
  - Type: list of strings
  - Default: None (will use column names if X is a DataFrame)

**Returns:**
- self (for method chaining)

**Example:**
```python
from sklearn.ensemble import RandomForestClassifier

# Single model
model = RandomForestClassifier()
xai.fit(model, X_train, y_train)

# Multiple models
models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'XGBoost': XGBClassifier()
}

xai.fit(models, X_train, y_train, feature_names=['feature1', 'feature2'])
```

#### 2. analyze()

```python
results = xai.analyze()
```

Performs comprehensive analysis of the fitted model(s).

**Returns:**
- Dictionary containing various analysis results:
  - 'model_performance': Metrics like accuracy, F1-score
  - 'feature_importance': Feature importance scores
  - 'shap_values': SHAP values for interpretability
  - 'cv_scores': Cross-validation scores
  - 'model_comparison': Comparison results if multiple models

**Example:**
```python
results = xai.analyze()
print("Model Performance:", results['model_performance'])
print("Top Features:", dict(list(results['feature_importance'].items())[:5]))
```

#### 3. predict()

```python
predictions = xai.predict(X)
```

Makes predictions using the fitted model.

**Parameters:**
- `X`: Features for prediction
  - Type: numpy array or pandas DataFrame

**Returns:**
- Predictions (class labels for classification, values for regression)
- Probabilities (for classification only)

**Example:**
```python
predictions, probabilities = xai.predict(X_test)
```

#### 4. explain_prediction()

```python
prediction, probabilities, explanation = xai.explain_prediction(input_data)
```

Provides detailed explanation for a single prediction.

**Parameters:**
- `input_data`: Single instance for prediction
  - Type: numpy array or pandas Series

**Returns:**
- prediction: The predicted value/class
- probabilities: Prediction probabilities (for classification)
- explanation: Detailed explanation from LLM

**Example:**
```python
input_data = X_test.iloc[0]
pred, prob, exp = xai.explain_prediction(input_data)
print(f"Prediction: {pred}")
print(f"Explanation: {exp}")
```

#### 5. generate_report()

```python
xai.generate_report(filename='xai_report.pdf')
```

Generates a comprehensive PDF report of the analysis.

**Parameters:**
- `filename`: Name of the output PDF file
  - Type: string
  - Default: 'xai_report.pdf'

**Example:**
```python
xai.generate_report('breast_cancer_analysis.pdf')
```

#### 6. perform_eda()

```python
XAIWrapper.perform_eda(df)
```

Performs exploratory data analysis on a dataset. This is a static method.

**Parameters:**
- `df`: Dataset to analyze
  - Type: pandas DataFrame

**Example:**
```python
XAIWrapper.perform_eda(my_dataset)
```

### Additional Module Functions

#### Fairness Module

```python
from explainableai.fairness import demographic_parity, equal_opportunity, disparate_impact
```

1. **demographic_parity()**
```python
max_diff, group_probs = demographic_parity(y_true, y_pred, protected_attribute)
```

2. **equal_opportunity()**
```python
max_diff, group_tpr = equal_opportunity(y_true, y_pred, protected_attribute)
```

3. **disparate_impact()**
```python
di_score, group_probs = disparate_impact(y_true, y_pred, protected_attribute)
```

**Example:**
```python
# Calculate fairness metrics
dp_diff, dp_probs = demographic_parity(y_true, y_pred, gender)
eo_diff, eo_tpr = equal_opportunity(y_true, y_pred, gender)
di_score, di_probs = disparate_impact(y_true, y_pred, gender)

# Plot fairness metrics
from explainableai.fairness import plot_fairness_metrics
fairness_metrics = {
    'Demographic Parity': dp_probs,
    'Equal Opportunity': eo_tpr
}
plot_fairness_metrics(fairness_metrics)
```

#### Feature Engineering Module

```python
from explainableai.feature_engineering import automated_feature_engineering
```

```python
X_train_eng, X_test_eng, feature_names = automated_feature_engineering(X_train, X_test)
```

#### Model Selection Module

```python
from explainableai.model_selection import get_default_models, compare_models
```

1. **get_default_models()**
```python
models = get_default_models()
```

2. **compare_models()**
```python
results = compare_models(X_train, y_train, X_test, y_test, models)
```

**Example:**
```python
# Get default models
models = get_default_models()

# Compare models
comparison_results = compare_models(X_train, y_train, X_test, y_test, models)

# Access results
for model_name, scores in comparison_results.items():
    print(f"{model_name}: CV Score = {scores['cv_score']:.4f}, Test Score = {scores['test_score']:.4f}")
```

### Utility Functions

#### Feature Selection

```python
from explainableai.feature_selection import select_features

X_selected, selected_indices = select_features(X, y, k=10)
```

#### Anomaly Detection

```python
from explainableai.anomaly_detection import detect_anomalies

anomalies = detect_anomalies(X)
```

### Complete Example

Here's a comprehensive example putting it all together:

```python
from explainableai import XAIWrapper
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# Initialize XAIWrapper
xai = XAIWrapper()

# Get default models
from explainableai.model_selection import get_default_models
models = get_default_models()

# Fit models
xai.fit(models, X_train, y_train, feature_names=data.feature_names)

# Analyze
results = xai.analyze()

# Make predictions
predictions, probabilities = xai.predict(X_test)

# Explain a single prediction
single_explanation = xai.explain_prediction(X_test[0])

# Generate report
xai.generate_report('breast_cancer_analysis.pdf')

# Fairness analysis (assuming we have a protected attribute)
from explainableai.fairness import demographic_parity
protected_attribute = np.random.binomial(1, 0.5, len(y_test))  # dummy protected attribute
dp_score, dp_probs = demographic_parity(y_test, predictions, protected_attribute)

# Print some results
print("Model Performance:", results['model_performance'])
print("Top Features:", dict(list(results['feature_importance'].items())[:5]))
print("Demographic Parity Score:", dp_score)
```