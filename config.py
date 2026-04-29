# ===============================
# CONFIGURATION FILE
# ===============================

# Model Configuration
MODEL_CONFIG = {
    "algorithm": "RandomForestClassifier",
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced"
}

# Training Configuration
TRAINING_CONFIG = {
    "test_size": 0.2,
    "stratify": True,
    "cross_validation_folds": 5,
    "random_state": 42
}

# File Paths
FILE_PATHS = {
    "train_data": "data/customer_churn_dataset-training-master.csv",
    "test_data": "data/customer_churn_dataset-testing-master.csv",
    "model": "churn_model.pkl",
    "encoders": "label_encoders.pkl",
    "predictions": "test_predictions.csv",
    "all_predictions": "all_customer_predictions.csv"
}

# Visualization Configuration
VIZ_CONFIG = {
    "figure_size": (12, 6),
    "dpi": 100,
    "style": "seaborn-v0_8-darkgrid",
    "colors": ["#2ecc71", "#e74c3c"]
}

# Prediction Thresholds
PREDICTION_CONFIG = {
    "churn_threshold": 0.5,
    "high_risk_threshold": 0.7
}
