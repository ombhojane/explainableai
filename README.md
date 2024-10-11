# ExplainableAI 🚀

<p align="center">
  <img src="https://raw.githubusercontent.com/alo7lika/explainableai/refs/heads/main/ExplainableAI%20-%20Model%20Insights.png" alt="Explainable AI Insights" width="70%">
</p>

[![PyPI version](https://img.shields.io/pypi/v/explainableai.svg)](https://pypi.org/project/explainableai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/explainableai.svg)](https://pypi.org/project/explainableai/)
[![Downloads](https://pepy.tech/badge/explainableai)](https://pepy.tech/project/explainableai)
[![GitHub stars](https://img.shields.io/github/stars/ombhojane/explainableai.svg)](https://github.com/ombhojane/explainableai/stargazers)

**ExplainableAI** is a powerful Python package that combines state-of-the-art machine learning techniques with advanced explainable AI methods and LLM-powered explanations. 🌟

---

## 🌟 Key Features

| Feature                              | Description                                                                                          |
|--------------------------------------|------------------------------------------------------------------------------------------------------|
| 📊 **Automated EDA**                 | Gain quick insights into your dataset.                                                               |
| 🧠 **Model Performance Evaluation**   | Comprehensive metrics for model assessment.                                                          |
| 📈 **Feature Importance Analysis**    | Understand which features drive your model's decisions.                                               |
| 🔍 **SHAP Integration**              | Deep insights into model behavior using SHAP (SHapley Additive exPlanations).                         |
| 📊 **Interactive Visualizations**     | Explore model insights through intuitive charts and graphs.                                           |
| 🤖 **LLM-Powered Explanations**       | Get human-readable explanations for model results and individual predictions.                         |
| 📑 **Automated Report Generation**    | Create professional PDF reports with a single command.                                                |
| 🔀 **Multi-Model Support**            | Compare and analyze multiple ML models simultaneously.                                                |
| ⚙️ **Easy-to-Use Interface**          | Simple API for model fitting, analysis, and prediction.                                               |

---

## 🚀 Quick Start

```bash
pip install explainableai
```
# ExplainableAI Example: Iris Dataset with Random Forest

## 📝 Code Overview

This example demonstrates how to use the `ExplainableAI` package to fit a Random Forest model on the Iris dataset, analyze model behavior, and generate an LLM-powered explanation and PDF report.

```python
from explainableai import XAIWrapper
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit model
xai = XAIWrapper()
model = RandomForestClassifier(n_estimators=100, random_state=42)
xai.fit(model, X_train, y_train)

# Analyze and explain results
results = xai.analyze(X_test, y_test)
print(results['llm_explanation'])

# Generate report
xai.generate_report('iris_analysis.pdf')
```

## 🛠️ Installation & Setup

Install ExplainableAI via pip:

```bash
pip install explainableai
```
To use LLM-powered explanations, you need to set up the following environment variable:

```makefile
GEMINI_API_KEY=your_api_key_here
```
# 🖥️ Usage Examples

## 🔀 Multi-Model Comparison

This example demonstrates how to compare multiple machine learning models using the ExplainableAI package.

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
### 🤖 Explaining Individual Predictions

```python
# After fitting the model

# New data to be explained
new_data = {'feature_1': value1, 'feature_2': value2, ...}  # Dictionary of feature values

# Make a prediction with explanation
prediction, probabilities, explanation = xai.explain_prediction(new_data)

print(f"Prediction: {prediction}")
print(f"Probabilities: {probabilities}")
print(f"Explanation: {explanation}")
```
## 📊 Feature Overview

| Module                | Description                                                                                     |
|-----------------------|-------------------------------------------------------------------------------------------------|
| `explore()`           | Automated exploratory data analysis (EDA) to uncover hidden insights.                         |
| `fit()`               | Train and analyze models with a simple API. Supports multiple models.                         |
| `analyze()`           | Evaluate model performance with SHAP and LLM-based explanations.                               |
| `explain_prediction()` | Explain individual predictions in plain English using LLMs.                                    |
| `generate_report()`    | Create professional PDF reports with visuals, explanations, and analysis.                     |

---

## 🌍 Running Locally

To run ExplainableAI locally:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ombhojane/explainableai.git
   cd explainableai
   ```
2.**Install Dependencies**:

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```
3.**Set up your environment variables**:

   Add your `GEMINI_API_KEY` to the `.env` file.

   ```bash
   GEMINI_API_KEY=your_api_key_here
```
### 🎥 Demo (Optional GIF)
**See ExplainableAI in action!** 

![Demo GIF](https://path-to-your-demo-gif.com/demo.gif) <!-- Replace with your actual GIF link -->

---

### 🤝 Contributing
We welcome contributions to ExplainableAI! Please check out our [Contributing Guidelines](CONTRIBUTING.md) to get started. Contributions are what make the open-source community an incredible place to learn, inspire, and create.

---

### 📄 License
ExplainableAI is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

### 🙌 Acknowledgements
ExplainableAI builds upon several open-source libraries, including:

- [scikit-learn](https://scikit-learn.org/)
- [SHAP](https://github.com/slundberg/shap)
- [Matplotlib](https://matplotlib.org/)
- [XGBoost](https://xgboost.readthedocs.io/)

Special thanks to all the contributors who have made this project possible!

<p align="center">
<a href="https://github.com/ombhojane/explainableai/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ombhojane/explainableai" alt="Contributors"/>
</a>
</p>




