# ===============================
# ENHANCED MODEL TRAINING SCRIPT
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import warnings

from config import MODEL_CONFIG, TRAINING_CONFIG, FILE_PATHS, VIZ_CONFIG
from utils import validate_data, encode_categorical_features, save_metrics_report
from model_evaluation import (
    calculate_metrics, print_detailed_report, plot_confusion_matrix,
    plot_roc_curve, plot_feature_importance, plot_precision_recall_curve
)

warnings.filterwarnings('ignore')

print("📌 Loading dataset...")
df = pd.read_csv(FILE_PATHS["train_data"])
print(f"✔ Dataset loaded! Shape: {df.shape}")

# ===============================
# DATA VALIDATION
# ===============================
print("\n📌 Validating data...")
issues = validate_data(df, required_columns=["Churn"])
if issues:
    print("⚠ Data issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✔ Data validation passed!")

# ===============================
# REMOVE MISSING TARGET VALUES
# ===============================
df = df[df["Churn"].notna()]
print(f"✔ Removed rows with missing Churn. Shape: {df.shape}")

# ===============================
# ENCODE CATEGORICAL COLUMNS
# ===============================
print("\n📌 Encoding categorical columns...")
df, encoders = encode_categorical_features(df, fit=True)
joblib.dump(encoders, FILE_PATHS["encoders"])
print(f"✔ Encoded {len(encoders)} categorical columns")
print(f"💾 Encoders saved to {FILE_PATHS['encoders']}")

# ===============================
# FEATURES & TARGET
# ===============================
X = df.drop("Churn", axis=1)
y = df["Churn"]

print(f"\nFeatures shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# ===============================
# TRAIN TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=TRAINING_CONFIG["test_size"],
    random_state=TRAINING_CONFIG["random_state"],
    stratify=y if TRAINING_CONFIG["stratify"] else None
)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ===============================
# TRAIN MODELS
# ===============================
models = {
    "RandomForest": RandomForestClassifier(**MODEL_CONFIG),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

results = {}

for model_name, model in models.items():
    print(f"\n📌 Training {model_name}...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    results[model_name] = {"model": model, "metrics": metrics}
    
    print(f"✔ {model_name} trained!")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")

# ===============================
# SELECT BEST MODEL
# ===============================
best_model_name = max(results.keys(), key=lambda x: results[x]["metrics"]["f1_score"])
best_model = results[best_model_name]["model"]

print(f"\n{'='*60}")
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"{'='*60}")

# ===============================
# DETAILED EVALUATION - BEST MODEL
# ===============================
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print_detailed_report(y_test, y_pred, best_model_name)

# ===============================
# CROSS VALIDATION
# ===============================
print(f"\n📌 Running cross-validation ({TRAINING_CONFIG['cross_validation_folds']}-fold)...")
cv_scores = cross_val_score(best_model, X_train, y_train, 
                            cv=TRAINING_CONFIG['cross_validation_folds'],
                            scoring='f1')
print(f"✔ CV Scores: {cv_scores}")
print(f"  Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ===============================
# SAVE BEST MODEL
# ===============================
joblib.dump(best_model, FILE_PATHS["model"])
print(f"\n💾 Best model saved to {FILE_PATHS['model']}")

# ===============================
# GENERATE VISUALIZATIONS
# ===============================
print("\n📊 Generating visualizations...")

plot_confusion_matrix(y_test, y_pred)
plot_roc_curve(y_test, y_pred_proba)
plot_precision_recall_curve(y_test, y_pred_proba)
plot_feature_importance(best_model, X.columns)

# ===============================
# SAVE METRICS REPORT
# ===============================
metrics_report = {
    "best_model": best_model_name,
    "accuracy": best_model.predict(X_test).tolist(),
    "model_metrics": results[best_model_name]["metrics"],
    "cross_validation": {
        "mean": float(cv_scores.mean()),
        "std": float(cv_scores.std()),
        "scores": cv_scores.tolist()
    },
    "train_set_size": len(X_train),
    "test_set_size": len(X_test)
}

save_metrics_report(results[best_model_name]["metrics"])

# ===============================
# CHURN DISTRIBUTION
# ===============================
plt.figure(figsize=VIZ_CONFIG["figure_size"])
y.value_counts().plot(kind="bar", color=VIZ_CONFIG["colors"])
plt.title("Training Data - Churn Distribution")
plt.xlabel("Churn (0=No, 1=Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("training_churn_distribution.png", dpi=VIZ_CONFIG["dpi"])
plt.close()
print("✔ Training churn distribution saved")

print("\n✅ Training completed successfully!")
print(f"📊 Model: {best_model_name}")
print(f"📈 F1-Score: {results[best_model_name]['metrics']['f1_score']:.4f}")
