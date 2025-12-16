# src/evaluate.py

import matplotlib.pyplot as plt
import numpy as np


from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)


#  EVALUATE MODEL 

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
    }

    results = {
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
    }

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        results["roc_curve"] = {"fpr": fpr, "tpr": tpr}

    return results



# CONFUSION MATRIX

def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix"):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    disp.plot(cmap="Blues", values_format="d", colorbar=False)
    plt.title(title)
    plt.grid(False)
    plt.show()



# SINGLE ROC CURVE

def plot_roc_curve(fpr, tpr, label=None):
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{label} (AUC={roc_auc:.3f})" if label else f"AUC={roc_auc:.3f}"
    )
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()



# MULTI-MODEL ROC

def plot_multiple_roc_curves(results_dict):
    """
    results_dict = {
        "Random Forest": (fpr, tpr),
        "XGBoost": (fpr, tpr)
    }
    """

    plt.figure(figsize=(8, 6))

    for model_name, (fpr, tpr) in results_dict.items():
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{model_name} (AUC={roc_auc:.3f})"
        )

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()



# PRECISION-RECALL CURVE

def plot_precision_recall(y_true, y_prob, model_name=None):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=f"{model_name} (AP={ap:.3f})" if model_name else f"AP={ap:.3f}"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()



# FEATURE IMPORTANCE

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Works for:
    - RandomForest
    - GradientBoosting
    - XGBoost
    """

    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not support feature_importances_")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(8, 6))
    plt.barh(
        range(len(indices)),
        importances[indices][::-1]
    )
    plt.yticks(
        range(len(indices)),
        [feature_names[i] for i in indices][::-1]
    )
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
    plt.grid(True)
    plt.show()
