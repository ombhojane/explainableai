# ExplainableAI User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Features](#core-features)
5. [Advanced Usage](#advanced-usage)
6. [API Reference](#api-reference)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Introduction

ExplainableAI is a powerful Python package designed to make machine learning models more interpretable and explainable. It provides a comprehensive suite of tools for model analysis, feature importance calculation, fairness assessment, and result visualization.

### Key Features
- Model-agnostic explanations
- Automated feature engineering
- Fairness metrics calculation
- Interactive visualizations
- Model comparison tools
- Comprehensive logging

## Installation

```bash
pip install explainableai
```

## Quick Start

Here's a simple example to get you started:

```python
from explainableai import XAIWrapper
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Initialize model and XAIWrapper
model = RandomForestClassifier(n_estimators=100, random_state=42)
xai = XAIWrapper()

# Fit the model
xai.fit(model, X_train, y_train, feature_names=data.feature_names)

# Analyze the model
results = xai.analyze()

# Generate a report
xai.generate_report('breast_cancer_analysis.pdf')
```

## Core Features

### 1. Model Analysis
The `XAIWrapper` class is the core component of the package. It provides:
- Model performance evaluation
- Feature importance calculation
- SHAP value computation
- Partial dependence plots

Example:
```python
# Get detailed model analysis
results = xai.analyze()
print(results['model_performance'])
```

### 2. Fairness Assessment
ExplainableAI includes tools for assessing model fairness:
- Demographic parity
- Equal opportunity
- Disparate impact

Example:
```python
from explainableai.fairness import demographic_parity

fairness_score, group_probs = demographic_parity(y_true, y_pred, protected_attribute)
```

### 3. Automated Feature Engineering
The package provides automated feature engineering capabilities:

```python
from explainableai.feature_engineering import automated_feature_engineering

X_train_engineered, X_test_engineered, new_feature_names = automated_feature_engineering(
    X_train, X_test
)
```

### 4. Model Comparison
Compare multiple models easily:

```python
from explainableai.model_selection import get_default_models, compare_models

models = get_default_models()
comparison_results = compare_models(X_train, y_train, X_test, y_test, models)
```

## Advanced Usage

### Handling Imbalanced Data
For imbalanced datasets, use the following approach:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score

model = RandomForestClassifier(class_weight='balanced')
xai = XAIWrapper()
xai.fit(model, X_train, y_train)
```

## API Reference

### XAIWrapper

#### Methods
- `fit(model, X, y, feature_names=None)`
- `analyze()`
- `predict(X)`
- `explain_prediction(input_data)`
- `generate_report(filename='xai_report.pdf')`
- `perform_eda(df)`

#### Attributes
- `model`: The fitted model
- `feature_importance`: Dictionary of feature importance scores
- `results`: Dictionary containing all analysis results

## Best Practices

1. **Data Preparation**
   - Always check for missing values
   - Normalize/standardize numerical features
   - Encode categorical variables appropriately

2. **Model Selection**
   - Start with simpler models before moving to complex ones
   - Use cross-validation for more reliable results

3. **Interpretation**
   - Don't rely on a single interpretation method
   - Consider both global and local explanations

## Troubleshooting

### Common Issues


1. **Slow performance with high-dimensional data**
   
   Solution: Use feature selection before analysis:
   ```python
   from explainableai.feature_selection import select_features
   X_selected, selected_indices = select_features(X, y, k=20)
   ```

### Logging

ExplainableAI uses Python's logging module. To see more detailed logs:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

We welcome contributions! Please see our [GitHub repository](https://github.com/ombhojane/explainableai/docs/Contributing.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/ombhojane/explainableai/LICENSE.md) file for details.