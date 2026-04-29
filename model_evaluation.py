# ===============================
# MODEL EVALUATION & ANALYSIS
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, auc
)
import seaborn as sns
import joblib
from config import VIZ_CONFIG

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate comprehensive evaluation metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_pred_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics

def print_detailed_report(y_true, y_pred, model_name="Model"):
    """
    Print detailed classification report
    """
    print(f"\n{'='*60}")
    print(f"DETAILED EVALUATION REPORT - {model_name}")
    print(f"{'='*60}\n")
    
    metrics = calculate_metrics(y_true, y_pred)
    
    print("Performance Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    """
    Plot and save confusion matrix
    """
    plt.figure(figsize=VIZ_CONFIG["figure_size"])
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(filename, dpi=VIZ_CONFIG["dpi"])
    plt.close()
    print(f"✔ Confusion matrix saved: {filename}")

def plot_roc_curve(y_true, y_pred_proba, filename="roc_curve.png"):
    """
    Plot and save ROC curve
    """
    plt.figure(figsize=VIZ_CONFIG["figure_size"])
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='#2ecc71', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#e74c3c', lw=2, linestyle='--', label='Random Classifier')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=VIZ_CONFIG["dpi"])
    plt.close()
    print(f"✔ ROC curve saved: {filename}")

def plot_precision_recall_curve(y_true, y_pred_proba, filename="precision_recall_curve.png"):
    """
    Plot and save Precision-Recall curve
    """
    plt.figure(figsize=VIZ_CONFIG["figure_size"])
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.plot(recall, precision, color='#3498db', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=VIZ_CONFIG["dpi"])
    plt.close()
    print(f"✔ Precision-Recall curve saved: {filename}")

def plot_feature_importance(model, feature_names, top_n=15, filename="feature_importance.png"):
    """
    Plot and save feature importance
    """
    if not hasattr(model, 'feature_importances_'):
        print("⚠ Model does not have feature_importances_ attribute")
        return
    
    importance_dict = dict(zip(feature_names, model.feature_importances_))
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    features, importances = zip(*sorted_features)
    
    plt.figure(figsize=(10, 8))
    plt.barh(features, importances, color='#2ecc71')
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    plt.savefig(filename, dpi=VIZ_CONFIG["dpi"])
    plt.close()
    print(f"✔ Feature importance saved: {filename}")
