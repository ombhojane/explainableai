import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init(autoreset=True)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
class XAIWrapper:
    def __init__(self):
        self.model = None
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
        self.results = None  # Add this line to store analysis results

    def fit(self, models, X, y, feature_names=None):
        logger.debug("Fitting the model...")
        try:
            if isinstance(models, dict):
                self.models = models
            else:
                self.models = {'Model': models}
            self.X = X
            self.y = y
            self.feature_names = feature_names if feature_names is not None else X.columns.tolist()
            self.is_classifier = all(hasattr(model, "predict_proba") for model in self.models.values())

            logger.info(f"{Fore.BLUE}Preprocessing data...{Style.RESET_ALL}")
            self._preprocess_data()

            logger.info(f"{Fore.BLUE}Fitting models and analyzing...{Style.RESET_ALL}")
            self.model_comparison_results = self._compare_models()

            # Select the best model based on cv_score
            best_model_name = max(self.model_comparison_results, key=lambda x: self.model_comparison_results[x]['cv_score'])
            self.model = self.models[best_model_name]
            self.model.fit(self.X, self.y)
            
            logger.info("Model fitting is complete...")
            return self
        except Exception as e:
            logger.error(f"Some error occur while fitting the models...{str(e)}")
    
    
    def _compare_models(self):
        logger.debug("Comparing the models...")
        try:
            from sklearn.model_selection import cross_val_score
            results = {}
            for name, model in self.models.items():
                cv_scores = cross_val_score(model, self.X, self.y, cv=5, scoring='roc_auc' if self.is_classifier else 'r2')
                model.fit(self.X, self.y)
                test_score = model.score(self.X, self.y)
                results[name] = {
                    'cv_score': cv_scores.mean(),
                    'test_score': test_score
                }
            logger.info("Comparing successfully...")
            return results
        except Exception as e:
            logger.error(f"Some error occur while comparing models...{str(e)}")

    def _preprocess_data(self):
        # Identify categorical and numerical columns
        self.categorical_columns = self.X.select_dtypes(include=['object', 'category']).columns
        self.numerical_columns = self.X.select_dtypes(include=['int64', 'float64']).columns

        # Create preprocessing steps
        logger.debug("Creating Preprocessing Steps...")
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
            ])
        logger.info("Pre proccessing completed...")

        # Fit and transform the data
        logger.debug("Fitting and transforming the data...")
        self.X = self.preprocessor.fit_transform(self.X)

        # Update feature names after preprocessing
        logger.debug("Updating feature names...")
        try:
            num_feature_names = self.numerical_columns.tolist()
            cat_feature_names = []
            if self.categorical_columns.size > 0:
                cat_feature_names = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_columns).tolist()
            self.feature_names = num_feature_names + cat_feature_names

            # Encode target variable if it's categorical
            if self.is_classifier and pd.api.types.is_categorical_dtype(self.y):
                self.label_encoder = LabelEncoder()
                self.y = self.label_encoder.fit_transform(self.y)
        except Exception as e:
            logger.error(f"Some error occur while updating...{str(e)}")

    def analyze(self):
        logger.debug("Analysing...")
        results = {}

        logger.info("Evaluating model performance...")
        results['model_performance'] = evaluate_model(self.model, self.X, self.y, self.is_classifier)

        logger.info("Calculating feature importance...")
        self.feature_importance = self._calculate_feature_importance()
        results['feature_importance'] = self.feature_importance

        logger.info("Generating visualizations...")
        self._generate_visualizations(self.feature_importance)

        logger.info("Calculating SHAP values...")
        results['shap_values'] = calculate_shap_values(self.model, self.X, self.feature_names)

        logger.info("Performing cross-validation...")
        mean_score, std_score = cross_validate(self.model, self.X, self.y)
        results['cv_scores'] = (mean_score, std_score)

        logger.info("Model comparison results:")
        results['model_comparison'] = self.model_comparison_results

        self._print_results(results)

        logger.info("Generating LLM explanation...")
        results['llm_explanation'] = get_llm_explanation(self.gemini_model, results)

        self.results = results
        return results
    
    def generate_report(self, filename='xai_report.pdf'):
        logger.debug("Report Generator...")
        try:
            if self.results is None:
                raise ValueError("No analysis results available. Please run analyze() first.")

            report = ReportGenerator(filename)
            report.add_heading("Explainable AI Report")

            report.add_heading("Model Comparison", level=2)
            model_comparison_data = [["Model", "CV Score", "Test Score"]]
            for model, scores in self.results['model_comparison'].items():
                model_comparison_data.append([model, f"{scores['cv_score']:.4f}", f"{scores['test_score']:.4f}"])
            report.add_table(model_comparison_data)


            # Model Performance
            report.add_heading("Model Performance", level=2)
            for metric, value in self.results['model_performance'].items():
                if isinstance(value, (int, float, np.float64)):
                    report.add_paragraph(f"**{metric}:** {value:.4f}")
                else:
                    report.add_paragraph(f"**{metric}:**\n{value}")

            # Feature Importance
            report.add_heading("Feature Importance", level=2)
            feature_importance_data = [["Feature", "Importance"]] + [[feature, f"{importance:.4f}"] for feature, importance in self.feature_importance.items()]
            report.add_table(feature_importance_data)

            # Visualizations
            report.add_heading("Visualizations", level=2)
            report.add_image('feature_importance.png')
            report.content.append(PageBreak())
            report.add_image('partial_dependence.png')
            report.content.append(PageBreak())
            report.add_image('learning_curve.png')
            report.content.append(PageBreak())
            report.add_image('correlation_heatmap.png')
            if self.is_classifier:
                report.content.append(PageBreak())
                report.add_image('roc_curve.png')
                report.content.append(PageBreak())
                report.add_image('precision_recall_curve.png')

            # LLM Explanation
            report.add_heading("LLM Explanation", level=2)
            report.add_llm_explanation(self.results['llm_explanation'])

            report.generate()
            logger.info("Report generated...")
        except Exception as e:
            logger.error(f"Some error occur in report generation...{str(e)}")

    def predict(self, X):
        logger.debug("Prediction...")
        try:
            if self.model is None:
                raise ValueError("Model has not been fitted. Please run fit() first.")
            
            X = self._preprocess_input(X)
            
            if self.is_classifier:
                prediction = self.model.predict(X)
                probabilities = self.model.predict_proba(X)
                if self.label_encoder:
                    prediction = self.label_encoder.inverse_transform(prediction)
                logger.info("Prediction Completed...")
                return prediction, probabilities
            else:
                prediction = self.model.predict(X)
                logger.info("Prediction Completed...")
                return prediction
        except Exception as e:
            logger.error(f"Error in prediction...{str(e)}")

    def _preprocess_input(self, X):
        # Ensure X is a DataFrame
        logger.debug("Preproceesing input...")
        try:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=self.feature_names)

            # Apply the same preprocessing as during training
            X = self.preprocessor.transform(X)
            logger.info("Preprocessing the data...")

            return X
        except Exception as e:
            logger.error(f"Some error occur in preprocessing the inpur...{str(e)}")

    def explain_prediction(self, input_data):
        logger.debug("Explaining the prediction...")
        input_df = pd.DataFrame([input_data])
        prediction, probabilities = self.predict(input_df)
        explanation = get_prediction_explanation(self.gemini_model, input_data, prediction[0], probabilities[0], self.feature_importance)
        logger.info("Prediction explained...")
        return prediction[0], probabilities[0], explanation
    
    def _calculate_feature_importance(self):
        logger.debug("Calculating the features...")
        perm_importance = permutation_importance(self.model, self.X, self.y, n_repeats=10, random_state=42)
        feature_importance = {feature: importance for feature, importance in zip(self.feature_names, perm_importance.importances_mean)}
        logger.info("Features calculated...")
        return dict(sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True))

    def _generate_visualizations(self, feature_importance):
        logger.debug("Generating visulatization...")
        try:
            plot_feature_importance(feature_importance)
            plot_partial_dependence(self.model, self.X, feature_importance, self.feature_names)
            plot_learning_curve(self.model, self.X, self.y)
            plot_correlation_heatmap(pd.DataFrame(self.X, columns=self.feature_names))
            if self.is_classifier:
                plot_roc_curve(self.model, self.X, self.y)
                plot_precision_recall_curve(self.model, self.X, self.y)
            logger.info("Visualizations generated.")
        except Exception as e:
            logger.error(f"Error in visulatization...{str(e)}")

    def _print_results(self, results):
        logger.debug("Printing results...")
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
            logger.error(f"Error occur in printing results...{str(e)}") 


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
            high_corr_list = [(corr_matrix.index[x], corr_matrix.columns[y]) for x, y in zip(*high_corr) if x != y and x < y]
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
            logger.error(f"Error occurred during exploratory data analysis...{str(e)}")
