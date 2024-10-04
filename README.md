# ExplainableAI

ExplainableAI is a Python package that provides tools for creating interpretable machine learning models. It combines various explainable AI techniques with LLM-powered explanations to make model predictions more understandable for both technical and non-technical users.
## Features

- Automated Exploratory Data Analysis (EDA)
- Model performance evaluation
- Feature importance calculation
- SHAP (SHapley Additive exPlanations) value calculation
- Visualization of model insights
- LLM-powered explanations of model results and individual predictions
- Automated report generation
- Easy-to-use interface for model fitting, analysis, and prediction

## Installation

You can install explainableai using pip:

```bash
  pip install explainableai
```

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`GEMINI_API_KEY`
## Usage/Examples


Here's a basic example of how to use ExplainableAI:

```python
from explainableai import XAIWrapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd

# Load your dataset
df = pd.read_csv('your_dataset.csv')
X = df.drop(columns=['target_column'])
y = df['target_column']

# Perform EDA
XAIWrapper.perform_eda(df)

# Create models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

# Create XAIWrapper instance
xai = XAIWrapper()

# Fit the models and run XAI analysis
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
```

## Run Locally

Clone the project

```bash
  git clone https://github.com/ombhojane/explainableai
```

Go to the project directory

```bash
  cd explainableai
```

Install dependencies

```bash
  pip install -r requirements.txt
```
Environment Values: Add Google's Gemini API key in .env as `GOOGLE_API_KEY`

Get Started with data.csv dataset or you can have any dataset

```bash
  python main.py [dataset] [Target_column]
```
## Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

Please adhere to this project's `code of conduct`.

## Credits

Explainable AI was created by [Om Bhojane](https://github.com/ombhojane). Special thanks to the following contributors for their support.

<p align="start">
<a  href="https://github.com/ombhojane/explainableai/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ombhojane/explainableai"/>
</a>
</p>

## Acknowledgements

- This package uses various open-source libraries including scikit-learn, shap, and matplotlib.
