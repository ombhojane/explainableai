# ExplainableAI: Comprehensive User Guide

    ExplainableAI is a Python package designed to enhance the interpretability of machine learning models. By combining explainable AI techniques with LLM-powered explanations, this package simplifies understanding of model predictions for both technical and non-technical users.

## Features:

- Automated Exploratory Data Analysis (EDA)
- Model performance evaluation
- Feature importance calculation
- SHAP (SHapley Additive exPlanations) value calculation
- Visualization of model insights
- LLM-powered explanations of model results and individual predictions
- Automated report generation
- Easy-to-use interface for model fitting, analysis, and prediction

## Installation
    To install ExplainableAI, use the following command in your terminal:

```bash
    pip install explainableai
```
## Environment Variables
    To run the project, ensure you add the required environment variables to your .env file:

`GEMINI_API_KEY`

### Quick Start Guide for New Users

    This section helps users quickly set up and start using the ExplainableAI package.
    Key Components:

    1. Introduction to ExplainableAI: A brief overview of the package, highlighting automated EDA, SHAP value calculations, and LLM-powered explanations.
    2. Installation Instructions: Step-by-step instructions for installing the package using pip or from the source, along with details on handling dependencies.
    3. Basic Usage Example: Demonstrates how to load data, train a model, and generate explanations with just a few lines of code.

Quick Code Example:
```python
        from explainableAI import XAIWrapper

        # Load data and train model
        explainer = XAIWrapper(model, X_train, y_train)

        # Fit the model using the training data
        explainer.fit()

        # Generate explanations for the model predictions
        explainer.analyze()

        # View LLM-powered explanation of predictions
        explainer.get_llm_explanation(predictions)

```

### API Documentation

This section provides an in-depth look at the available classes and methods within ExplainableAI, including example code for each.

Key Classes and Methods:

1. XAIWrapper:
    Purpose: Central class for fitting models and generating explanations.
    Key Methods:
        fit(self): Trains the model and prepares it for analysis.
        analyze(self): Generates SHAP values and other explainable AI outputs.
        get_llm_explanation(self, predictions): Uses LLMs to provide human-readable explanations for individual model predictions.
        generate_report(self, output_path): Automatically generates a report on the model’s performance and explainability results.

Code Example:

    ```python
        explainer = XAIWrapper(model, X_train, y_train)
        explainer.fit()
        explainer.analyze()
    ```

        2. EDAExplainer:
        Purpose: Automates the Exploratory Data Analysis process.
        Key Methods:
        perform_eda(self, data): Runs EDA on the input dataset.
        visualize_correlations(self, data): Generates correlation heatmaps and other visual insights.

Code Example:

```python
    eda_explainer = EDAExplainer(data)
    eda_explainer.perform_eda()
    eda_explainer.visualize_correlations()
```

## In-Depth Tutorials

    Detailed tutorials guide users through various use cases and help them maximize the benefits of ExplainableAI.

#    Tutorial 1: Automated EDA and Visualization
        Objective: Demonstrate how to perform EDA and visualize insights.
        Steps:
                    1.   Load the data and initialize EDAExplainer.
                    2.   Perform EDA and generate visualizations like correlation heatmaps and distribution plots.
                    3.     Interpret the insights generated.

Code Example:
```python
        from explainableAI import EDAExplainer
        # Perform EDA
        eda = EDAExplainer(data)
        eda.perform_eda()

        # Visualize insights
        eda.visualize_correlations()
```
#       Tutorial 2: Model Performance Evaluation & SHAP Value Calculation
            Objective: Evaluate model performance and calculate feature importance     using       SHAP values.
            Steps:
                        Train a model using the ModelExplainer.
                        Evaluate the model’s performance with metrics such as accuracy or confusion matrix.
                        Calculate SHAP values to understand feature importance.
Code Example:
```python
        from explainableAI import ModelExplainer

        # Fit and explain the model
        explainer = ModelExplainer(model, X_train, y_train)
        explainer.explain_model()

        # Plot SHAP values
        explainer.plot_shap_summary()
```
#        Tutorial 3: LLM-Powered Explanations
            Objective: Showcase how ExplainableAI uses LLMs to provide natural  language      explanations of model predictions.
            Steps:
                1. Train a model and make predictions.
                2. Use the LLM-powered explanation feature to translate predictions into natural language.
Code Example:
```bash
      # Get natural language explanation
      explainer.get_llm_explanation(predictions)
```
#   Automated Report Generation
    This feature allows users to generate comprehensive reports detailing model performance and explainability.
    Key Steps for Generating Reports:
```python
    explainer.generate_report(output_path='model_report.pdf')
```

#   Best Practices and Tips
        To use ExplainableAI effectively, consider the following best practices:
    Choosing the Right Model: Select models that balance accuracy and interpretability.
    SHAP Explanations: Generate SHAP values for explaining feature importance clearly.
    LLM-Powered Explanations: Utilize natural language explanations in presentations or stakeholder reports for better understanding.
