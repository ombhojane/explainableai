# explainableai/core.py

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
        if isinstance(models, dict):
            self.models = models
        else:
            self.models = {'Model': models}
        self.X = X
        self.y = y
        self.feature_names = feature_names if feature_names is not None else X.columns.tolist()
        self.is_classifier = all(hasattr(model, "predict_proba") for model in self.models.values())

        print("Preprocessing data...")
        self._preprocess_data()

        print("Fitting models and analyzing...")
        self.model_comparison_results = self._compare_models()
        
        # Select the best model based on cv_score
        best_model_name = max(self.model_comparison_results, key=lambda x: self.model_comparison_results[x]['cv_score'])
        self.model = self.models[best_model_name]
        self.model.fit(self.X, self.y)
        
        return self
    
    def _compare_models(self):
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
        return results

    def _preprocess_data(self):
        # Identify categorical and numerical columns
        self.categorical_columns = self.X.select_dtypes(include=['object', 'category']).columns
        self.numerical_columns = self.X.select_dtypes(include=['int64', 'float64']).columns

        # Create preprocessing steps
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

        # Fit and transform the data
        self.X = self.preprocessor.fit_transform(self.X)

        # Update feature names after preprocessing
        num_feature_names = self.numerical_columns.tolist()
        cat_feature_names = []
        if self.categorical_columns.size > 0:
            cat_feature_names = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_columns).tolist()
        self.feature_names = num_feature_names + cat_feature_names

        # Encode target variable if it's categorical
        if self.is_classifier and pd.api.types.is_categorical_dtype(self.y):
            self.label_encoder = LabelEncoder()
            self.y = self.label_encoder.fit_transform(self.y)

    def analyze(self):
        results = {}

        print("Evaluating model performance...")
        results['model_performance'] = evaluate_model(self.model, self.X, self.y, self.is_classifier)

        print("Calculating feature importance...")
        self.feature_importance = self._calculate_feature_importance()
        results['feature_importance'] = self.feature_importance

        print("Generating visualizations...")
        self._generate_visualizations(self.feature_importance)

        print("Calculating SHAP values...")
        results['shap_values'] = calculate_shap_values(self.model, self.X, self.feature_names)

        print("Performing cross-validation...")
        mean_score, std_score = cross_validate(self.model, self.X, self.y)
        results['cv_scores'] = (mean_score, std_score)

        print("Model comparison results:")
        results['model_comparison'] = self.model_comparison_results

        self._print_results(results)

        print("Generating LLM explanation...")
        results['llm_explanation'] = get_llm_explanation(self.gemini_model, results)

        self.results = results
        return results
    
    def generate_report(self, filename='xai_report.pdf'):
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

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been fitted. Please run fit() first.")
        
        X = self._preprocess_input(X)
        
        if self.is_classifier:
            prediction = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            if self.label_encoder:
                prediction = self.label_encoder.inverse_transform(prediction)
            return prediction, probabilities
        else:
            prediction = self.model.predict(X)
            return prediction

    def _preprocess_input(self, X):
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)

        # Apply the same preprocessing as during training
        X = self.preprocessor.transform(X)

        return X

    def explain_prediction(self, input_data):
        input_df = pd.DataFrame([input_data])
        prediction, probabilities = self.predict(input_df)
        explanation = get_prediction_explanation(self.gemini_model, input_data, prediction[0], probabilities[0], self.feature_importance)
        return prediction[0], probabilities[0], explanation
    
    def _calculate_feature_importance(self):
        perm_importance = permutation_importance(self.model, self.X, self.y, n_repeats=10, random_state=42)
        feature_importance = {feature: importance for feature, importance in zip(self.feature_names, perm_importance.importances_mean)}
        return dict(sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True))

    def _generate_visualizations(self, feature_importance):
        plot_feature_importance(feature_importance)
        plot_partial_dependence(self.model, self.X, feature_importance, self.feature_names)
        plot_learning_curve(self.model, self.X, self.y)
        plot_correlation_heatmap(pd.DataFrame(self.X, columns=self.feature_names))
        if self.is_classifier:
            plot_roc_curve(self.model, self.X, self.y)
            plot_precision_recall_curve(self.model, self.X, self.y)

    def _print_results(self, results):
        print("\nModel Performance:")
        for metric, value in results['model_performance'].items():
            if isinstance(value, (int, float, np.float64)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}:\n{value}")

        print("\nTop 5 Important Features:")
        for feature, importance in list(results['feature_importance'].items())[:5]:
            print(f"{feature}: {importance:.4f}")

        print(f"\nCross-validation Score: {results['cv_scores'][0]:.4f} (+/- {results['cv_scores'][1]:.4f})")

        print("\nVisualizations saved:")
        print("- Feature Importance: feature_importance.png")
        print("- Partial Dependence: partial_dependence.png")
        print("- Learning Curve: learning_curve.png")
        print("- Correlation Heatmap: correlation_heatmap.png")
        if self.is_classifier:
            print("- ROC Curve: roc_curve.png")
            print("- Precision-Recall Curve: precision_recall_curve.png")

        if results['shap_values'] is not None:
            print("\nSHAP values calculated successfully. See 'shap_summary.png' for visualization.")
        else:
            print("\nSHAP values calculation failed. Please check the console output for more details.") 


    @staticmethod
    def perform_eda(df):
        print("\nExploratory Data Analysis:")
        print(f"Dataset shape: {df.shape}")
        print("\nDataset info:")
        df.info()
        print("\nSummary statistics:")
        print(df.describe())
        print("\nMissing values:")
        print(df.isnull().sum())
        print("\nData types:")
        print(df.dtypes)
        print("\nUnique values in each column:")
        for col in df.columns:
            print(f"{col}: {df[col].nunique()}")
        
        # Additional EDA steps
        print("\nCorrelation matrix:")
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        print(corr_matrix)
        
        # Identify highly correlated features
        high_corr = np.where(np.abs(corr_matrix) > 0.8)
        high_corr_list = [(corr_matrix.index[x], corr_matrix.columns[y]) for x, y in zip(*high_corr) if x != y and x < y]
        if high_corr_list:
            print("\nHighly correlated features:")
            for feat1, feat2 in high_corr_list:
                print(f"{feat1} - {feat2}: {corr_matrix.loc[feat1, feat2]:.2f}")
        
        # Identify potential outliers
        print("\nPotential outliers (values beyond 3 standard deviations):")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            outliers = df[(df[col] < mean - 3*std) | (df[col] > mean + 3*std)]
            if not outliers.empty:
                print(f"{col}: {len(outliers)} potential outliers")
        
        # Class distribution for the target variable (assuming last column is target)
        target_col = df.columns[-1]
        print(f"\nClass distribution for target variable '{target_col}':")
        print(df[target_col].value_counts(normalize=True))