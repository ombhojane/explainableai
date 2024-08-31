import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from .utils import explain_model, calculate_metrics
from .visualizations import plot_feature_importance, plot_partial_dependence, plot_interactive_feature_importance
from .feature_analysis import calculate_shap_values
from .feature_engineering import automated_feature_engineering
from .model_selection import compare_models
from .anomaly_detection import detect_anomalies

def xai_wrapper(train_function):
    def wrapper(*args, **kwargs):
        print("Starting XAI wrapper...")
        
        X_train, y_train = args[0], args[1]
        X_test, y_test = kwargs.get('X_test'), kwargs.get('y_test')
        
        # Automated feature engineering
        X_train_fe, X_test_fe, feature_names = automated_feature_engineering(X_train, X_test)
        
        start_time = time.time()
        model = train_function(X_train_fe, y_train)
        end_time = time.time()
        
        training_time = end_time - start_time
        print(f"Model training completed in {training_time:.2f} seconds.")
        
        explanation = explain_model(model, X_train_fe, y_train, X_test_fe, y_test, feature_names)
        metrics = calculate_metrics(model, X_test_fe, y_test)
        
        # Generate visualizations
        plot_feature_importance(explanation['feature_importance'])
        plot_partial_dependence(model, X_train_fe, explanation['feature_importance'], feature_names)
        plot_interactive_feature_importance(explanation['feature_importance'])
        
        # Calculate SHAP values
        try:
            shap_values = calculate_shap_values(model, X_train_fe, feature_names)
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            shap_values = None
        
        # Model comparison
        model_comparison = compare_models(X_train_fe, y_train, X_test_fe, y_test)
        
        # Anomaly detection
        anomalies = detect_anomalies(X_test_fe)
        
        return model, {
            'explanation': explanation,
            'metrics': metrics,
            'shap_values': shap_values,
            'model_comparison': model_comparison,
            'anomalies': anomalies,
            'feature_names': feature_names,
            'X_train_fe': X_train_fe,
            'X_test_fe': X_test_fe
        }
    
    return wrapper

def analyze_dataset(file_path):
    df = pd.read_csv(file_path)
    print("Dataset shape:", df.shape)
    print("\nSample data:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nSummary statistics:")
    print(df.describe())
    return df

def preprocess_data(df, target_column):
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Handle categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y