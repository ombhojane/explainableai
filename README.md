# ExplainableAI

[![PyPI version](https://img.shields.io/pypi/v/explainableai.svg)](https://pypi.org/project/explainableai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/explainableai.svg)](https://pypi.org/project/explainableai/)
[![Downloads](https://pepy.tech/badge/explainableai)](https://pepy.tech/project/explainableai)
[![GitHub stars](https://img.shields.io/github/stars/ombhojane/explainableai.svg)](https://github.com/ombhojane/explainableai/stargazers)

ExplainableAI is a powerful Python package that combines state-of-the-art machine learning techniques with advanced explainable AI methods and LLM-powered explanations.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Running Locally](#running-locally)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Features

- **Automated Exploratory Data Analysis (EDA)**: Gain quick insights into your dataset.
- **Model Performance Evaluation**: Comprehensive metrics for model assessment.
- **Feature Importance Analysis**: Understand which features drive your model's decisions.
- **SHAP (SHapley Additive exPlanations) Integration**: Deep insights into model behavior.
- **Interactive Visualizations**: Explore model insights through intuitive charts and graphs.
- **LLM-Powered Explanations**: Get human-readable explanations for model results and individual predictions.
- **Automated Report Generation**: Create professional PDF reports with a single command.
- **Multi-Model Support**: Compare and analyze multiple ML models simultaneously.
- **Easy-to-Use Interface**: Simple API for model fitting, analysis, and prediction.

## Installation

Install ExplainableAI using pip:

```bash
pip install explainableai
```

## Quick Start

```python
from explainableai import XAIWrapper
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load sample dataset
X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XAIWrapper
xai = XAIWrapper()

# Fit and analyze model
model = RandomForestClassifier(n_estimators=100, random_state=42)
xai.fit(model, X_train, y_train)
results = xai.analyze(X_test, y_test)

# Print LLM explanation
print(results['llm_explanation'])

# Generate report
xai.generate_report('iris_analysis.pdf')
```

## Usage Examples

### Multi-Model Comparison

```python
from explainableai import XAIWrapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pandas as pd

# Load your dataset
df = pd.read_csv('your_dataset.csv')
X = df.drop(columns=['target_column'])
y = df['target_column']

# Create models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42)
}

# Initialize XAIWrapper
xai = XAIWrapper()

# Fit and analyze models
xai.fit(models, X, y)
results = xai.analyze()

# Print LLM explanation of results
print(results['llm_explanation'])

# Generate a comprehensive report
xai.generate_report('multi_model_comparison.pdf')
```

### Explaining Individual Predictions

```python
# ... (after fitting the model)

# Make a prediction with explanation
new_data = {...}  # Dictionary of feature values
prediction, probabilities, explanation = xai.explain_prediction(new_data)

print(f"Prediction: {prediction}")
print(f"Probabilities: {probabilities}")
print(f"Explanation: {explanation}")
```

## Environment Variables

To use the LLM-powered explanations, you need to set up the following environment variable:

- `GEMINI_API_KEY`: Your [Google Gemini API key](https://ai.google.dev/gemini-api/docs/api-key)

Add this to your `.env` file:

```
GEMINI_API_KEY=your_api_key_here
```

## API Reference

For detailed API documentation, please refer to our [API Reference](https://pypi.org/project/explainableai/).

## Running Locally

To run ExplainableAI locally:

1. Clone the repository:

   ```bash
   git clone https://github.com/ombhojane/explainableai.git
   cd explainableai
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables (see [Environment Variables](#environment-variables)).

4. Run the example script:
   ```bash
   python main.py [dataset] [target_column]
   ```

## Contributing

We welcome contributions to ExplainableAI! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more information on how to get started.

## Credits

Explainable AI was created by [Om Bhojane](https://github.com/ombhojane). Special thanks to the following contributors for their support.

<p align="start">
<a  href="https://github.com/ombhojane/explainableai/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ombhojane/explainableai"/>
</a>
</p>

## Acknowledgements

ExplainableAI builds upon several open-source libraries, including:

- [scikit-learn](https://scikit-learn.org/)
- [SHAP](https://github.com/slundberg/shap)
- [Matplotlib](https://matplotlib.org/)
- [XGBoost](https://xgboost.readthedocs.io/)

We are grateful to the maintainers and contributors of these projects.

## License

ExplainableAI is released under the [MIT License](LICENSE).
