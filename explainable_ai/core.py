# core.py

import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from .utils import explain_model, calculate_metrics
from .visualizations import plot_feature_importance, plot_partial_dependence, plot_interactive_feature_importance
from .feature_analysis import calculate_shap_values
from .feature_engineering import automated_feature_engineering
from .model_selection import compare_models
from .anomaly_detection import detect_anomalies
from .feature_selection import select_features
from .model_evaluation import cross_validate, plot_learning_curve, plot_roc_curve, plot_precision_recall_curve, plot_calibration_curve
from .feature_interaction import analyze_feature_interactions
import time
import pandas as pd
import numpy as np

def xai_wrapper(train_function):
    def wrapper(X, y, **kwargs):
        print("Starting XAI wrapper...")
        
        test_size = kwargs.get('test_size', 0.2)
        random_state = kwargs.get('random_state', 42)
        
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        print("Performing automated feature engineering...")
        start_time = time.time()
        X_train_fe, X_test_fe, feature_names = automated_feature_engineering(X_train, X_test)
        print(f"Feature engineering completed in {time.time() - start_time:.2f} seconds.")
        
        print("Performing feature selection...")
        start_time = time.time()
        X_train_selected, selected_indices = select_features(X_train_fe, y_train)
        X_test_selected = X_test_fe[:, selected_indices]
        selected_feature_names = [feature_names[i] for i in selected_indices]
        print(f"Feature selection completed in {time.time() - start_time:.2f} seconds.")
        
        print("Training model...")
        start_time = time.time()
        model = train_function(X_train_selected, y_train)
        training_time = time.time() - start_time
        print(f"Model training completed in {training_time:.2f} seconds.")
        
        print("Explaining model...")
        start_time = time.time()
        explanation = explain_model(model, X_train_selected, y_train, X_test_selected, y_test, selected_feature_names)
        print(f"Model explanation completed in {time.time() - start_time:.2f} seconds.")
        
        print("Calculating metrics...")
        metrics = calculate_metrics(model, X_test_selected, y_test)
        
        print("Generating visualizations...")
        plot_feature_importance(explanation['feature_importance'])
        plot_partial_dependence(model, X_train_selected, explanation['feature_importance'], selected_feature_names)
        plot_interactive_feature_importance(explanation['feature_importance'])
        
        print("Calculating SHAP values...")
        start_time = time.time()
        try:
            shap_values = calculate_shap_values(model, X_train_selected, selected_feature_names)
            print(f"SHAP values calculation completed in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            shap_values = None
        
        print("Comparing models...")
        start_time = time.time()
        model_comparison = compare_models(X_train_selected, y_train, X_test_selected, y_test)
        print(f"Model comparison completed in {time.time() - start_time:.2f} seconds.")
        
        print("Detecting anomalies...")
        start_time = time.time()
        anomalies = detect_anomalies(X_test_selected)
        print(f"Anomaly detection completed in {time.time() - start_time:.2f} seconds.")
        
        print("Performing cross-validation...")
        start_time = time.time()
        cv_score, cv_std = cross_validate(model, X_train_selected, y_train)
        print(f"Cross-validation completed in {time.time() - start_time:.2f} seconds.")
        
        print("Generating evaluation plots...")
        plot_learning_curve(model, X_train_selected, y_train)
        plot_roc_curve(model, X_test_selected, y_test)
        plot_precision_recall_curve(model, X_test_selected, y_test)
        plot_calibration_curve(model, X_test_selected, y_test)
        
        print("Analyzing feature interactions...")
        start_time = time.time()
        interactions = analyze_feature_interactions(model, X_train_selected, selected_feature_names)
        print(f"Feature interaction analysis completed in {time.time() - start_time:.2f} seconds.")
        
        print("XAI wrapper completed.")
        return model, {
            'explanation': explanation,
            'metrics': metrics,
            'shap_values': shap_values,
            'model_comparison': model_comparison,
            'anomalies': anomalies,
            'feature_names': selected_feature_names,
            'X_train_selected': X_train_selected,
            'X_test_selected': X_test_selected,
            'y_train': y_train,
            'y_test': y_test,
            'cv_score': cv_score,
            'cv_std': cv_std,
            'interactions': interactions
        }
    
    return wrapper

def analyze_dataset(data):
    print("Dataset shape:", data.shape)
    print("\nSample data:")
    print(data.head())
    print("\nData types:")
    print(data.dtypes)
    print("\nSummary statistics:")
    print(data.describe())
    return data


def preprocess_data(data, target_column):
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Handle categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y

