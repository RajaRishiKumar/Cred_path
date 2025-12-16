# src/inference.py

import joblib
import pandas as pd

def load_artifacts(model_path, artifacts_path):
    model = joblib.load(model_path)
    artifacts = joblib.load(artifacts_path)
    return model, artifacts


def run_inference(df, model, artifacts):
    """
    artifacts = {
        "scaler": fitted_scaler,
        "numeric_cols": [...],
        "feature_order": [...]
    }
    """

    df = df.copy()

    # Ensure same feature order
    df = df.reindex(columns=artifacts["feature_order"], fill_value=0)

    # Scale ONLY numeric columns
    numeric_cols = artifacts["numeric_cols"]
    scaler = artifacts["scaler"]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1] if hasattr(model, "predict_proba") else None

    return preds, probs

def save_predictions(df, preds, probs=None, output_path="predictions.csv"):
    df = df.copy()
    df["prediction"] = preds
    if probs is not None:
        df["probability"] = probs
    df.to_csv(output_path, index=False)
    return df