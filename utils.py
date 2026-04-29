# ===============================
# UTILITY FUNCTIONS
# ===============================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from datetime import datetime

def validate_data(df, required_columns=None):
    """
    Validate dataset structure and quality
    """
    issues = []
    
    # Check for empty dataframe
    if df.empty:
        issues.append("Dataset is empty")
        return issues
    
    # Check for required columns
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            issues.append(f"Missing columns: {missing}")
    
    # Check for all NaN columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        issues.append(f"Columns with all missing values: {null_cols}")
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate rows")
    
    return issues

def handle_missing_values(df, strategy="drop"):
    """
    Handle missing values in dataset
    
    Args:
        df: DataFrame
        strategy: 'drop' or 'fill'
    """
    if strategy == "drop":
        return df.dropna()
    elif strategy == "fill":
        # Fill numeric with mean, categorical with mode
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "UNKNOWN", inplace=True)
        return df
    return df

def encode_categorical_features(df, encoders=None, fit=False):
    """
    Encode categorical features
    
    Args:
        df: DataFrame
        encoders: Dictionary of LabelEncoders (if None, creates new ones)
        fit: Whether to fit new encoders
    """
    df_copy = df.copy()
    new_encoders = {} if fit else {}
    
    for col in df_copy.select_dtypes(include=["object"]).columns:
        if fit:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
            new_encoders[col] = le
        elif encoders and col in encoders:
            le = encoders[col]
            df_copy[col] = df_copy[col].astype(str)
            # Handle unseen values
            mask = df_copy[col].isin(le.classes_)
            df_copy.loc[~mask, col] = le.classes_[0]  # Use first class for unseen
            df_copy[col] = le.transform(df_copy[col])
    
    return df_copy, new_encoders if fit else encoders

def get_feature_importance_dict(model, feature_names):
    """
    Get feature importance as dictionary
    """
    if hasattr(model, 'feature_importances_'):
        importance = dict(zip(feature_names, model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    return {}

def create_backup(filename):
    """
    Create backup of file with timestamp
    """
    if os.path.exists(filename):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{filename}.backup_{timestamp}"
        os.rename(filename, backup_name)
        print(f"✔ Backup created: {backup_name}")

def save_metrics_report(metrics_dict, filename="model_metrics_report.txt"):
    """
    Save metrics to text file
    """
    with open(filename, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("MODEL PERFORMANCE METRICS\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in metrics_dict.items():
            if isinstance(value, dict):
                f.write(f"\n{key}:\n")
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"✔ Metrics report saved: {filename}")
