# src/evaluate.py

import logging

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    recall_score,
    precision_score,
)

logger = logging.getLogger("ml_pipeline")


# -------------------------------------------------
# 🔹 Business Loss Function (vectorized)
# -------------------------------------------------

def expected_loss(
    y,
    probs,
    threshold: float,
    config: dict,
) -> float:
    """
    Computes mean expected business loss per applicant.

    Approved (low-risk) = probs < threshold
      • If actual default  → loss of loan_amt
      • If actual good     → gain of profit_amt  (negative loss)

    Parameters
    ----------
    y       : array-like (pandas Series or numpy array)
    probs   : array-like
    threshold : float
    config  : full config dict (reads config["business"])

    Returns
    -------
    float — mean loss per applicant
    """
    # Coerce to numpy to avoid bitwise-& issues between bool arrays and pd.Series
    y_arr     = np.asarray(y, dtype=int)
    probs_arr = np.asarray(probs, dtype=float)

    loan   = config["business"]["loan_amt"]
    profit = config["business"]["profit_amt"]

    approve = probs_arr < threshold

    loss = (
        (approve & (y_arr == 1)) * loan     # approved defaulters → loss
        - (approve & (y_arr == 0)) * profit  # approved good payers → profit
    )

    return float(loss.mean())


# -------------------------------------------------
# 🔹 Threshold Optimisation
# -------------------------------------------------

def find_best_threshold(y, probs, config: dict):
    """
    Grid-searches over [threshold.min, threshold.max] in threshold.steps
    steps and returns the threshold that minimises expected_loss.

    Returns
    -------
    best_t   : float
    best_loss: float
    """
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

    logger.info(f"Best threshold: {best_t:.4f}  |  Expected loss: {best_loss:.4f}")
    return best_t, best_loss


# -------------------------------------------------
# 🔹 Evaluation Metrics
# -------------------------------------------------

def evaluate_model(
    y,
    probs,
    threshold: float,
    max_samples: int = 5000,
) -> None:
    """
    Prints confusion matrix and classification report.

    Parameters
    ----------
    y           : array-like
    probs       : array-like
    threshold   : decision threshold (prob > threshold → predicted default)
    max_samples : if len(y) > max_samples, a random subsample is used
                  to avoid OOM on large test sets (mirrors notebook behaviour).
                  Set to None to evaluate on the full set.
    """
    y_arr     = np.asarray(y, dtype=int)
    probs_arr = np.asarray(probs, dtype=float)

    # Optional subsampling (matches notebook's 5000-row sample)
    if max_samples is not None and len(y_arr) > max_samples:
        rng = np.random.default_rng(seed=42)
        idx      = rng.choice(len(y_arr), size=max_samples, replace=False)
        y_arr     = y_arr[idx]
        probs_arr = probs_arr[idx]
        logger.info(
            f"evaluate_model: subsampled to {max_samples} rows "
            f"(pass max_samples=None to evaluate on full test set)"
        )

    preds = (probs_arr > threshold).astype(int)

    logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_arr, preds)))
    logger.info(
        "Classification Report:\n" + classification_report(y_arr, preds)
    )

    recall    = recall_score(y_arr, preds)
    precision = precision_score(y_arr, preds)

    logger.info(f"Recall (Default):    {recall:.4f}")
    logger.info(f"Precision (Default): {precision:.4f}")