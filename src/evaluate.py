# src/evaluate.py

import logging
import json
import joblib
import yaml
import pandas as pd
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    recall_score,
    precision_score,
    roc_auc_score,
)

logger = logging.getLogger("ml_pipeline")


# -------------------------------------------------
# 🔹 Business Loss Function (unchanged)
# -------------------------------------------------

def expected_loss(y, probs, threshold: float, config: dict) -> float:
    y_arr = np.asarray(y, dtype=int)
    probs_arr = np.asarray(probs, dtype=float)

    loan   = config["business"]["loan_amt"]
    profit = config["business"]["profit_amt"]

    approve = probs_arr < threshold

    loss = (
        (approve & (y_arr == 1)) * loan
        - (approve & (y_arr == 0)) * profit
    )

    return float(loss.mean())


# -------------------------------------------------
# 🔹 Threshold Optimisation (unchanged)
# -------------------------------------------------

def find_best_threshold(y, probs, config: dict):
    thresholds = np.linspace(
        config["threshold"]["min"],
        config["threshold"]["max"],
        config["threshold"]["steps"],
    )

    best_t    = 0.5
    best_loss = float("inf")

    for t in thresholds:
        loss = expected_loss(y, probs, t, config)
        if loss < best_loss:
            best_loss = loss
            best_t    = t

    logger.info(f"Best threshold: {best_t:.4f}  |  Loss: {best_loss:.4f}")
    return best_t, best_loss


# -------------------------------------------------
# 🔹 Evaluation (unchanged)
# -------------------------------------------------

def evaluate_model(y, probs, threshold: float):
    y_arr = np.asarray(y, dtype=int)
    probs_arr = np.asarray(probs, dtype=float)

    preds = (probs_arr > threshold).astype(int)

    logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_arr, preds)))
    logger.info("Classification Report:\n" + classification_report(y_arr, preds))

    recall    = recall_score(y_arr, preds)
    precision = precision_score(y_arr, preds)

    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"Precision: {precision:.4f}")

    return recall, precision


# -------------------------------------------------
# 🔹 MAIN (DVC ENTRY POINT)
# -------------------------------------------------

def main():

    logger.info("Running evaluation stage …")

    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Load test data
    test = pd.read_csv("data/test.csv")

    y_test = test["TARGET"]
    X_test = test.drop(["TARGET", "SK_ID_CURR"], axis=1)

    # Load model
    model = joblib.load("artifacts/model.pkl")

    # Predict
    probs = model.predict_proba(X_test)[:, 1]

    # Threshold optimisation
    best_t, best_loss = find_best_threshold(y_test, probs, config)

    # Evaluation metrics
    recall, precision = evaluate_model(y_test, probs, best_t)
    auc = roc_auc_score(y_test, probs)

    # Save metrics (CRITICAL for DVC)
    metrics = {
        "auc": float(auc),
        "recall": float(recall),
        "precision": float(precision),
        "best_threshold": float(best_t),
        "business_loss": float(best_loss),
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    logger.info("Saved metrics.json")


if __name__ == "__main__":
    main()