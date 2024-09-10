# ExplainableAI

ExplainableAI is a Python package that provides tools for creating interpretable machine learning models. It combines various explainable AI techniques with LLM-powered explanations to make model predictions more understandable for both technical and non-technical users.
## Features

- Automated Exploratory Data Analysis (EDA)
- Model performance evaluation
- Feature importance calculation
- Feature interaction analysis
- SHAP (SHapley Additive exPlanations) value calculation
- Visualization of model insights
- LLM-powered explanations of model results and individual predictions
- Cross-validation
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
import pandas as pd

# Load your dataset
df = pd.read_csv('your_dataset.csv')
X = df.drop(columns=['target_column'])
y = df['target_column']

# Perform EDA
XAIWrapper.perform_eda(df)

# Create and initialize your model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Create XAIWrapper instance
xai = XAIWrapper()

# Fit the model and run XAI analysis
xai.fit(model, X, y, feature_names=X.columns.tolist())
results = xai.analyze()

# Print LLM explanation of results
print(results['llm_explanation'])

# Make a prediction with explanation
new_data = {...}  # Dictionary of feature values
prediction, probabilities, explanation = xai.explain_prediction(new_data)
print(f"Prediction: {prediction}")
print(f"Probabilities: {probabilities}")
print(f"Explanation: {explanation}")
```
## Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

Please adhere to this project's `code of conduct`.


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

Get Started with data.csv dataset or you can have any dataset

```bash
  python main.py [dataset] [Target_column]
```


## Acknowledgements

- This package uses various open-source libraries including scikit-learn, shap, and matplotlib.

