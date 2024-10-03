# ExplainableAI: Comprehensive User Guide

ExplainableAI is a Python package designed to enhance the interpretability of machine learning models. By combining explainable AI techniques with LLM-powered explanations, this package simplifies understanding of model predictions for both technical and non-technical users.

## Features:

- Automated Exploratory Data Analysis (EDA)
      This feature helps users automatically explore the structure and relationships within their datasets. It visualizes important statistical summaries, correlations, and         
      distributions to give a quick understanding of the data.
      Example:
```python
        from explainableAI import EDAExplainer
        eda = EDAExplainer(data)
        eda.perform_eda()  # Automatically performs EDA
        eda.visualize_correlations()  # Generates a correlation heatmap
```
     Key Functions: perform_eda(), visualize_correlations()
     Benefit: Saves time and ensures a thorough examination of data without manually writing code for common EDA tasks.
  
- Model performance evaluation
          This feature helps users evaluate the performance of machine learning models by calculating common metrics like accuracy, precision, recall, and F1-score. It also includes 
          confusion matrix plots to show how well the model is performing.
          Example:
```python
        from explainableAI import ModelEvaluator
        evaluator = ModelEvaluator(model, X_test, y_test)
        evaluator.evaluate_model()  # Evaluate model performance
```
        Key Functions: evaluate_model()
        Benefit: Provides a comprehensive evaluation of the model's strengths and weaknesses using standard metrics.
- Feature importance calculation
          This feature calculates which features contribute most to model predictions, typically using SHAP (SHapley Additive exPlanations) values, which quantify how much each feature 
          influences the model output.
          Example:
```python
        from explainableAI import ModelExplainer
        explainer = ModelExplainer(model, X_train, y_train)
        explainer.plot_shap_summary()  # Plot feature importance with SHAP values
```
        Key Functions: plot_shap_summary()
        Benefit: Allows users to see the importance of individual features in an interpretable way, helping them understand the model's decision-making process.
        
- SHAP (SHapley Additive exPlanations) value calculation
          SHAP values are a powerful method for explaining individual predictions. They decompose a prediction into the contributions of each feature, making it easy to see why a model 
          made a particular decision.
          Example:
```python
            explainer.explain_model()  # Automatically computes SHAP values
            explainer.plot_shap_values()  # Visualize SHAP values for specific instances
```
Key Functions: explain_model(), plot_shap_values()
Benefit: Provides a transparent explanation of model predictions, making it easy to see how changes in feature values affect the output.

- Visualization of model insights
      This feature generates visualizations to help users understand the data and model predictions, such as correlation heatmaps, feature importance plots, and SHAP value summaries.
      Example:
```python
        from explainableAI import DataVisualizer
        visualizer = DataVisualizer(data)
        visualizer.plot_shap_summary()  # Visualize SHAP value summary for features
```
        Key Functions: plot_shap_summary(), visualize_correlations()
        Benefit: Visual representations of data and model performance make it easier to communicate insights to others.

- LLM-powered explanations of model results and individual predictions
          ExplainableAI leverages large language models (LLMs) to provide natural language explanations of complex model predictions. This feature translates technical insights into 
          human-readable summaries.
          Example:
```python
        predictions = model.predict(X_test)
        explainer.get_llm_explanation(predictions)  # LLM provides human-readable explanations
```
        Key Functions: get_llm_explanation()
        Benefit: This is particularly useful when explaining model predictions to non-technical stakeholders, making AI decisions more accessible and understandable.
- Automated report generation
      This feature automatically generates a comprehensive report, summarizing the model's performance, feature importance, SHAP values, and LLM-generated explanations. The report can 
      be output in PDF or HTML formats.
      Example:
```python
        explainer.generate_report(output_path='report.pdf')  # Generate a report with all insights
```
        Key Functions: generate_report()
        Benefit: Saves time by automating the reporting process and providing detailed insights in a professional format that can be shared with stakeholders.
- Easy-to-use interface for model fitting, analysis, and prediction
          Ensures the machine learning model is fair by checking for biases in its predictions and providing a fairness report. This feature is critical for applications where fairness 
          and bias must be carefully monitored, such as hiring or lending.
          Example:
```python
            from explainableAI import FairnessAnalyzer
            fairness_analyzer = FairnessAnalyzer(model, data)
            fairness_analyzer.generate_fairness_report()  # Evaluate fairness of the model
```
Key Functions: generate_fairness_report()
            Benefit: Helps identify and mitigate potential biases in machine learning models, ensuring more ethical AI applications.


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




# Directory Structure
    Files
        __init__.py: Initializes the package.
             Example Code -
``` Python
                 from .anomaly_detection import AnomalyDetector
                 from .llm_explanations import LLMExplainer
                 from .model_evaluation import ModelEvaluator
```    

        anomaly_detection.py: Implements anomaly detection methods.
            Example Code -
``` Python
                class AnomalyDetector:
                    def __init__(self, model, data):
                        self.model = model
                        self.data = data
                
                    def detect_anomalies(self):
                        # Fit the model and detect anomalies
                        anomalies = self.model.fit_predict(self.data)
                        return anomalies
```
        core.py: Contains core functions for the project.
            Example Code -
```Python
                def load_data(filepath):
                import pandas as pd
                data = pd.read_csv(filepath)
                return data
 ```
        fairness.py: Ensures model fairness with preprocessing and report generation.
            Example Code - 
```python
                class FairnessAnalyzer:
                    def __init__(self, model, data):
                        self.model = model
                        self.data = data
                
                    def generate_fairness_report(self):
                        # Logic to check for fairness in model predictions
                        return fairness_report
```
        feature_analysis.py: Analyzes dataset features for insights.
            Example Code -
```python
                class FeatureAnalyzer:
                    def __init__(self, data):
                        self.data = data
                
                    def analyze_features(self):
                        # Logic for feature analysis
                        feature_importances = ...
                        return feature_importances
 ```
        feature_engineering.py: Performs transformations to enhance feature quality.
            Example Code - 
```python
                class FeatureEngineer:
                    def __init__(self, data):
                        self.data = data
                
                    def transform_features(self):
                        # Logic for feature engineering
                        engineered_features = ...
                        return engineered_features
 ```
        feature_interaction.py: Examines interactions between dataset features.
            Example Code - 
```python
                class FeatureInteractionAnalyzer:
                    def __init__(self, data):
                        self.data = data
                
                    def analyze_interactions(self):
                        # Logic to examine feature interactions
                        interactions = ...
                        return interactions
 ```
        feature_selection.py: Selects important features for model training.
            Example Code -
```python
                class FeatureSelector:
                    def __init__(self, model, data):
                        self.model = model
                        self.data = data
                
                    def select_features(self):
                        # Logic for feature selection
                        selected_features = ...
                        return selected_features
```
        llm_explanations.py: Provides explanations using large language models.
            Example Code - 
```python
                class LLMExplainer:
                    def __init__(self, model):
                        self.model = model
                
                    def get_llm_explanation(self, predictions):
                        # Use LLM to generate natural language explanations
                        explanation = ...
                        return explanation
```
        model_comparison.py: Compares the performance of different models.
            Example Code -
```python
                class ModelComparer:
                    def __init__(self, models, data):
                        self.models = models
                        self.data = data
                
                    def compare_models(self):
                        # Logic to compare models
                        model_performance = ...
                        return model_performance
 ```
        model_evaluation.py: Evaluates models with various performance metrics.
            Example Code - 
```python
                class ModelEvaluator:
                    def __init__(self, model, X_test, y_test):
                        self.model = model
                        self.X_test = X_test
                        self.y_test = y_test
 ```
        model_interpretability.py: Enhances model transparency and interpretability.
            Example Code -
```python
                class ModelInterpretability:
                    def __init__(self, model, data):
                        self.model = model
                        self.data = data
                
                    def interpret_model(self):
                        # Logic to explain model predictions
                        explanations = ...
                        return explanations
 ```
        model_selection.py: Manages the selection of optimal machine learning models.
            Example Code - 
```python
                class ModelSelector:
                    def __init__(self, data):
                        self.data = data
                
                    def select_best_model(self):
                        # Logic for model selection
                        best_model = ...
                        return best_model
 ```
        report_generator.py: Generates detailed and user-friendly reports.
            Example Code -
```python
                class ReportGenerator:
                    def __init__(self, model, data, report_path):
                        self.model = model
                        self.data = data
                        self.report_path = report_path
                
                    def generate_report(self):
                        # Logic to generate reports
                        report = ...
                        return report
```
        utils.py: Contains utility functions used throughout the project.
            Example Code -
```python
                def preprocess_data(data):
                    # Logic to clean and preprocess data
                    processed_data = ...
                    return processed_data
```
        visualizations.py: Creates visualizations for data and model performance.
            Example Code -
```python
                class DataVisualizer:
                def __init__(self, data):
                    self.data = data
            
                def plot_shap_summary(self):
                    # Logic to create SHAP summary plot
                    plot = ...
                    return plot
```
            
