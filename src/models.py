# src/models.py

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False


# ------------------ MODEL REGISTRY ------------------

def get_model_registry(random_state=42):
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced"
        ),
        "naive_bayes": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=random_state
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=random_state
        ),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "svm_linear": SVC(kernel="linear", probability=True, class_weight="balanced"),
        "svm_rbf": SVC(kernel="rbf", probability=True, class_weight="balanced"),
    }

    if xgb_available:
        models["xgboost"] = XGBClassifier(
            eval_metric="logloss", random_state=random_state
        )

    

    return models


# ------------------ TRAIN MODEL ------------------

def train_model(
    X_train,
    y_train,
    model_name,
    params=None,
    use_smote=False,
    random_state=42
):
    registry = get_model_registry(random_state)

    if model_name not in registry:
        raise ValueError(f"Unknown model: {model_name}")

    model = registry[model_name]

    if model_name == "xgboost":
        X_train = X_train.values
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)




    if params:
        model.set_params(**params)

    if use_smote:
        if y_train.nunique() < 2:
            raise ValueError(
             "SMOTE requested but y_train has only one class. "
            "Check preprocessing or split."
            )

        sm = SMOTE(random_state=random_state)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    model.fit(X_train, y_train)
    return model


