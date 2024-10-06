# model_interpretability.py

# Import colorama and its components
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init(autoreset=True)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import TensorFlow
import tensorflow as tf
from scikeras.wrappers import KerasClassifier, KerasRegressor

from .visualizations import (
    plot_feature_importance, plot_partial_dependence, plot_learning_curve,
    plot_roc_curve, plot_precision_recall_curve, plot_correlation_heatmap
)
from .model_evaluation import evaluate_model, cross_validate
from .feature_analysis import calculate_shap_values
from .feature_interaction import analyze_feature_interactions
from .llm_explanations import initialize_gemini, get_llm_explanation, get_prediction_explanation
from .report_generator import ReportGenerator
from .model_selection import compare_models
from reportlab.platypus import PageBreak
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class XAIWrapper:
    def __init__(self):
        self.model = None
        self.models = {}
        self.X = None
        self.y = None
        self.feature_names = None
        self.is_classifier = None
        self.preprocessor = None
        self.label_encoder = None
        self.categorical_columns = None
        self.numerical_columns = None
        self.gemini_model = initialize_gemini()
        self.feature_importance = None
        self.results = None
        self.model_type = None  # To store model type

    def fit(self, models, X, y, feature_names=None):
        logger.debug("Starting the fit process...")
        try:
            # Initialize models
            if isinstance(models, dict):
                self.models = models
                logger.debug("Initialized models from dictionary input.")
            else:
                self.models = {'Model': models}
                logger.debug("Initialized single model.")
            
            self.X = X
            self.y = y
            self.feature_names = feature_names if feature_names is not None else X.columns.tolist()
            self._determine_model_type()

            logger.info(f"{Fore.BLUE}Preprocessing data...{Style.RESET_ALL}")
            self._preprocess_data()

            logger.info(f"{Fore.BLUE}Fitting models and analyzing...{Style.RESET_ALL}")
            self.model_comparison_results = self._compare_models()

            # Select the best model based on cv_score
            best_model_name = max(
                self.model_comparison_results, 
                key=lambda x: self.model_comparison_results[x]['cv_score']
            )
            self.model = self.models[best_model_name]
            logger.info(f"Selected best model: {best_model_name} with CV Score: {self.model_comparison_results[best_model_name]['cv_score']:.4f}")

            # Fit the selected model
            if self.model_type == 'tensorflow':
                logger.info("Fitting TensorFlow model...")
                self.model.fit(self.X, self.y, epochs=10, batch_size=32, verbose=0)
            else:
                logger.info("Fitting scikit-learn model...")
                self.model.fit(self.X, self.y)
            
            logger.info("Model fitting is complete.")
            return self
        except Exception as e:
            logger.error(f"An error occurred while fitting the models: {str(e)}")
            raise

    def _determine_model_type(self):
        logger.debug("Determining model type...")
        try:
            model_types = set()
            for model in self.models.values():
                if isinstance(model, (tf.keras.Model, KerasClassifier, KerasRegressor)):
                    model_types.add('tensorflow')
                else:
                    model_types.add('sklearn')
            if len(model_types) > 1:
                raise ValueError("All models should be of the same type (either all TensorFlow or all scikit-learn).")
            self.model_type = model_types.pop()
            logger.debug(f"Detected model type: {self.model_type}")

            # Determine if models are classifiers
            if self.model_type == 'tensorflow':
                # Assume TensorFlow models output probabilities for classifiers
                self.is_classifier = all(
                    model.output_shape[-1] > 1 for model in self.models.values()
                )
            else:
                self.is_classifier = all(hasattr(model, "predict_proba") for model in self.models.values())
            logger.debug(f"Is classifier: {self.is_classifier}")
        except Exception as e:
            logger.error(f"Error determining model type: {str(e)}")
            raise

    def _compare_models(self):
        logger.debug("Comparing models...")
        try:
            results = {}
            for name, model in self.models.items():
                logger.debug(f"Evaluating model: {name}")
                if self.model_type == 'tensorflow':
                    # Wrap TensorFlow models for scikit-learn compatibility
                    if self.is_classifier:
                        wrapped_model = KerasClassifier(build_fn=lambda: model, epochs=10, batch_size=32, verbose=0)
                    else:
                        wrapped_model = KerasRegressor(build_fn=lambda: model, epochs=10, batch_size=32, verbose=0)
                    
                    cv_scores = cross_validate(
                        wrapped_model, 
                        self.X, 
                        self.y, 
                        is_classifier=self.is_classifier, 
                        model_type=self.model_type
                    )
                    test_score = wrapped_model.score(self.X, self.y)
                else:
                    # Determine scoring metric
                    scoring = 'roc_auc' if self.is_classifier else 'r2'
                    cv_scores = cross_val_score(model, self.X, self.y, cv=5, scoring=scoring)
                    model.fit(self.X, self.y)
                    test_score = model.score(self.X, self.y)
                
                results[name] = {
                    'cv_score': np.mean(cv_scores),
                    'test_score': test_score
                }
                logger.debug(f"Model {name}: CV Score = {results[name]['cv_score']:.4f}, Test Score = {results[name]['test_score']:.4f}")
            logger.info("Model comparison completed successfully.")
            return results
        except Exception as e:
            logger.error(f"An error occurred while comparing models: {str(e)}")
            raise

    def _preprocess_data(self):
        logger.debug("Preprocessing data...")
        try:
            # Identify categorical and numerical columns
            self.categorical_columns = self.X.select_dtypes(include=['object', 'category']).columns
            self.numerical_columns = self.X.select_dtypes(include=['int64', 'float64']).columns
            logger.debug(f"Categorical columns: {list(self.categorical_columns)}")
            logger.debug(f"Numerical columns: {list(self.numerical_columns)}")

            # Create preprocessing pipelines
            logger.debug("Creating preprocessing pipelines...")
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, self.numerical_columns),
                    ('cat', categorical_transformer, self.categorical_columns)
                ]
            )
            logger.info("Preprocessing pipelines created.")

            # Fit and transform the data
            logger.debug("Fitting and transforming the data...")
            self.X = self.preprocessor.fit_transform(self.X)
            logger.info("Data preprocessing completed.")

            # Update feature names after preprocessing
            logger.debug("Updating feature names post-preprocessing...")
            try:
                num_feature_names = self.numerical_columns.tolist()
                cat_feature_names = []
                if len(self.categorical_columns) > 0:
                    cat_feature_names = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_columns).tolist()
                self.feature_names = num_feature_names + cat_feature_names
                logger.debug(f"Updated feature names: {self.feature_names}")

                # Encode target variable if it's categorical
                if self.is_classifier and pd.api.types.is_categorical_dtype(self.y):
                    self.label_encoder = LabelEncoder()
                    self.y = self.label_encoder.fit_transform(self.y)
                    logger.debug("Encoded target variable using LabelEncoder.")
            except Exception as e:
                logger.error(f"Error updating feature names: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Error during data preprocessing: {str(e)}")
            raise

    def analyze(self):
        logger.debug("Starting analysis...")
        results = {}
        try:
            # Evaluate model performance
            logger.info("Evaluating model performance...")
            results['model_performance'] = evaluate_model(
                self.model, self.X, self.y, self.is_classifier, self.model_type
            )

            # Calculate feature importance
            logger.info("Calculating feature importance...")
            self.feature_importance = self._calculate_feature_importance()
            results['feature_importance'] = self.feature_importance

            # Generate visualizations
            logger.info("Generating visualizations...")
            self._generate_visualizations(self.feature_importance)

            # Calculate SHAP values
            logger.info("Calculating SHAP values...")
            results['shap_values'] = calculate_shap_values(
                self.model, self.X, self.feature_names, self.model_type
            )

            # Perform cross-validation
            logger.info("Performing cross-validation...")
            mean_score, std_score = cross_validate(
                self.model, self.X, self.y, 
                is_classifier=self.is_classifier, 
                model_type=self.model_type
            )
            results['cv_scores'] = (mean_score, std_score)

            # Add model comparison results
            logger.info("Adding model comparison results...")
            results['model_comparison'] = self.model_comparison_results

            # Print results
            self._print_results(results)

            # Generate LLM explanation
            logger.info("Generating LLM explanation...")
            results['llm_explanation'] = get_llm_explanation(self.gemini_model, results)

            self.results = results
            logger.debug("Analysis completed successfully.")
            return results
        except Exception as e:
            logger.error(f"An error occurred during analysis: {str(e)}")
            raise

    def generate_report(self, filename='xai_report.pdf'):
        logger.debug("Generating report...")
        if self.results is None:
            raise ValueError("No analysis results available. Please run analyze() first.")

        try:
            report = ReportGenerator(filename)
            report.add_heading("Explainable AI Report")

            sections = {
                'model_comparison': self._generate_model_comparison,
                'model_performance': self._generate_model_performance,
                'feature_importance': self._generate_feature_importance,
                'visualization': self._generate_visualization,
                'llm_explanation': self._generate_llm_explanation
            }

            if input("Do you want all sections in the XAI report? (y/n) ").strip().lower() in ['y', 'yes']:
                for section_func in sections.values():
                    section_func(report)
            else:
                for section, section_func in sections.items():
                    if input(f"Do you want {section} in the XAI report? (y/n) ").strip().lower() in ['y', 'yes']:
                        section_func(report)

            report.generate()
            logger.info(f"Report generated successfully and saved as '{filename}'.")
        except Exception as e:
            logger.error(f"An error occurred while generating the report: {str(e)}")
            raise

    def _generate_model_comparison(self, report):
        logger.debug("Adding model comparison section to report...")
        report.add_heading("Model Comparison", level=2)
        model_comparison_data = [["Model", "CV Score", "Test Score"]] + [
            [model, f"{scores['cv_score']:.4f}", f"{scores['test_score']:.4f}"]
            for model, scores in self.results['model_comparison'].items()
        ]
        report.add_table(model_comparison_data)
        logger.debug("Model comparison section added.")

    def _generate_model_performance(self, report):
        logger.debug("Adding model performance section to report...")
        report.add_heading("Model Performance", level=2)
        for metric, value in self.results['model_performance'].items():
            if isinstance(value, (int, float, np.float64)):
                report.add_paragraph(f"**{metric}:** {value:.4f}")
            else:
                report.add_paragraph(f"**{metric}:**\n{value}")
        logger.debug("Model performance section added.")

    def _generate_feature_importance(self, report):
        logger.debug("Adding feature importance section to report...")
        report.add_heading("Feature Importance", level=2)
        feature_importance_data = [["Feature", "Importance"]] + [
            [feature, f"{importance:.4f}"] for feature, importance in self.feature_importance.items()
        ]
        report.add_table(feature_importance_data)
        logger.debug("Feature importance section added.")

    def _generate_visualization(self, report):
        logger.debug("Adding visualizations section to report...")
        report.add_heading("Visualizations", level=2)
        visualization_files = [
            'feature_importance.png', 'partial_dependence.png',
            'learning_curve.png', 'correlation_heatmap.png'
        ]
        if self.is_classifier:
            visualization_files += ['roc_curve.png', 'precision_recall_curve.png']
        
        for image in visualization_files:
            report.add_image(image)
            report.content.append(PageBreak())
        logger.debug("Visualizations section added.")

    def _generate_llm_explanation(self, report):
        logger.debug("Adding LLM explanation section to report...")
        report.add_heading("LLM Explanation", level=2)
        report.add_llm_explanation(self.results['llm_explanation'])
        logger.debug("LLM explanation section added.")

    def predict(self, X):
        logger.debug("Starting prediction...")
        try:
            if self.model is None:
                raise ValueError("Model has not been fitted. Please run fit() first.")
            
            X_preprocessed = self._preprocess_input(X)
            
            if self.is_classifier:
                prediction = self.model.predict(X_preprocessed)
                probabilities = self.model.predict_proba(X_preprocessed)
                if self.label_encoder:
                    prediction = self.label_encoder.inverse_transform(prediction)
                logger.info("Prediction completed successfully.")
                return prediction, probabilities
            else:
                prediction = self.model.predict(X_preprocessed)
                logger.info("Prediction completed successfully.")
                return prediction
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def _preprocess_input(self, X):
        logger.debug("Preprocessing input data for prediction...")
        try:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=self.feature_names)
                logger.debug("Converted input to DataFrame.")
            
            # Apply the same preprocessing as during training
            X_preprocessed = self.preprocessor.transform(X)
            logger.debug("Input data preprocessed successfully.")
            return X_preprocessed
        except Exception as e:
            logger.error(f"Error during input preprocessing: {str(e)}")
            raise

    def explain_prediction(self, input_data):
        logger.debug("Generating prediction explanation...")
        try:
            input_df = pd.DataFrame([input_data])
            prediction, probabilities = self.predict(input_df)
            explanation = get_prediction_explanation(
                self.gemini_model,
                input_data,
                prediction[0],
                probabilities[0],
                self.feature_importance
            )
            logger.info("Prediction explanation generated successfully.")
            return prediction[0], probabilities[0], explanation
        except Exception as e:
            logger.error(f"Error during prediction explanation: {str(e)}")
            raise

    def _calculate_feature_importance(self):
        logger.debug("Calculating feature importance...")
        try:
            if self.model_type == 'tensorflow':
                logger.debug("Calculating SHAP values for TensorFlow model...")
                shap_values = calculate_shap_values(
                    self.model, self.X, self.feature_names, self.model_type
                )
                feature_importance = np.mean(np.abs(shap_values.values), axis=0)
                feature_importance_dict = {
                    feature: importance 
                    for feature, importance in zip(self.feature_names, feature_importance)
                }
                logger.debug("SHAP-based feature importance calculated.")
            else:
                logger.debug("Calculating permutation importance for scikit-learn model...")
                perm_importance = permutation_importance(
                    self.model, self.X, self.y, n_repeats=10, random_state=42
                )
                feature_importance_dict = {
                    feature: importance 
                    for feature, importance in zip(self.feature_names, perm_importance.importances_mean)
                }
                logger.debug("Permutation-based feature importance calculated.")
            
            # Sort features by absolute importance in descending order
            sorted_importance = dict(
                sorted(feature_importance_dict.items(), key=lambda item: abs(item[1]), reverse=True)
            )
            self.feature_importance = sorted_importance
            logger.info("Feature importance calculated and sorted.")
            return sorted_importance
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            raise

    def _generate_visualizations(self, feature_importance):
        logger.debug("Generating visualizations...")
        try:
            plot_feature_importance(feature_importance)
            plot_partial_dependence(
                self.model, self.X, feature_importance, self.feature_names, self.model_type
            )
            plot_learning_curve(
                self.model, self.X, self.y, self.is_classifier, self.model_type
            )
            plot_correlation_heatmap(
                pd.DataFrame(self.X, columns=self.feature_names)
            )
            if self.is_classifier:
                plot_roc_curve(
                    self.model, self.X, self.y, self.model_type
                )
                plot_precision_recall_curve(
                    self.model, self.X, self.y, self.model_type
                )
            logger.info("Visualizations generated and saved successfully.")
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise

    def _print_results(self, results):
        logger.debug("Printing analysis results...")
        try:
            logger.info("\nModel Performance:")
            for metric, value in results['model_performance'].items():
                if isinstance(value, (int, float, np.float64)):
                    logger.info(f"{metric}: {value:.4f}")
                else:
                    logger.info(f"{metric}:\n{value}")

            logger.info("\nTop 5 Important Features:")
            for feature, importance in list(results['feature_importance'].items())[:5]:
                logger.info(f"{feature}: {importance:.4f}")

            logger.info(f"\nCross-validation Score: {results['cv_scores'][0]:.4f} (+/- {results['cv_scores'][1]:.4f})")

            logger.info("\nVisualizations saved:")
            logger.info("- Feature Importance: feature_importance.png")
            logger.info("- Partial Dependence: partial_dependence.png")
            logger.info("- Learning Curve: learning_curve.png")
            logger.info("- Correlation Heatmap: correlation_heatmap.png")
            if self.is_classifier:
                logger.info("- ROC Curve: roc_curve.png")
                logger.info("- Precision-Recall Curve: precision_recall_curve.png")

            if results['shap_values'] is not None:
                logger.info("\nSHAP values calculated successfully. See 'shap_summary.png' for visualization.")
            else:
                logger.info("\nSHAP values calculation failed. Please check the console output for more details.")
        except Exception as e:
            logger.error(f"Error printing results: {str(e)}")
            raise

    @staticmethod
    def perform_eda(df):
        logger.debug("Performing exploratory data analysis...")
        try:
            logger.info(f"{Fore.CYAN}Exploratory Data Analysis:{Style.RESET_ALL}")
            logger.info(f"{Fore.GREEN}Dataset shape: {df.shape}{Style.RESET_ALL}")
            logger.info(f"{Fore.CYAN}Dataset info:{Style.RESET_ALL}")
            df.info()
            logger.info(f"{Fore.CYAN}Summary statistics:{Style.RESET_ALL}")
            logger.info(df.describe())
            logger.info(f"{Fore.CYAN}Missing values:{Style.RESET_ALL}")
            logger.info(df.isnull().sum())
            logger.info(f"{Fore.CYAN}Data types:{Style.RESET_ALL}")
            logger.info(df.dtypes)
            logger.info(f"{Fore.CYAN}Unique values in each column:{Style.RESET_ALL}")
            for col in df.columns:
                logger.info(f"{Fore.GREEN}{col}: {df[col].nunique()}{Style.RESET_ALL}")

            # Additional EDA steps
            logger.info(f"{Fore.CYAN}Correlation matrix:{Style.RESET_ALL}")
            corr_matrix = df.select_dtypes(include=[np.number]).corr()
            logger.info(corr_matrix)

            # Identify highly correlated features
            high_corr = np.where(np.abs(corr_matrix) > 0.8)
            high_corr_list = [
                (corr_matrix.index[x], corr_matrix.columns[y]) 
                for x, y in zip(*high_corr) 
                if x != y and x < y
            ]
            if high_corr_list:
                logger.info(f"{Fore.YELLOW}Highly correlated features:{Style.RESET_ALL}")
                for feat1, feat2 in high_corr_list:
                    logger.info(f"{Fore.GREEN}{feat1} - {feat2}: {corr_matrix.loc[feat1, feat2]:.2f}{Style.RESET_ALL}")

            # Identify potential outliers
            logger.info(f"{Fore.CYAN}Potential outliers (values beyond 3 standard deviations):{Style.RESET_ALL}")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                mean = df[col].mean()
                std = df[col].std()
                outliers = df[(df[col] < mean - 3 * std) | (df[col] > mean + 3 * std)]
                if not outliers.empty:
                    logger.info(f"{Fore.GREEN}{col}: {len(outliers)} potential outliers{Style.RESET_ALL}")

            # Class distribution for the target variable (assuming last column is target)
            target_col = df.columns[-1]
            logger.info(f"{Fore.CYAN}Class distribution for target variable '{target_col}':{Style.RESET_ALL}")
            logger.info(df[target_col].value_counts(normalize=True))
        except Exception as e:
            logger.error(f"Error occurred during exploratory data analysis: {str(e)}")
            raise 

