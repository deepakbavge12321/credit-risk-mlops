# src/model.py

import logging

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger("ml_pipeline")


# -------------------------------------------------
# 🔹 Train model with K-Fold CV
# -------------------------------------------------

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    feature_names: list,
    config: dict,
) -> dict:
    """
    Trains a LightGBM classifier with stratified K-fold CV.

    Returns
    -------
    dict with keys:
        model            – last-fold LGBMClassifier (used for calibration)
        oof_preds        – out-of-fold probability predictions on train set
        test_preds       – averaged probability predictions on test set
        scores           – per-fold AUC list
        oof_auc          – overall OOF AUC
        feature_importance – DataFrame with per-fold importances
    """
    # Class imbalance handling
    scale_pos_weight = (len(y) - y.sum()) / y.sum()

    # Build params — merge model block + training block items LightGBM cares about
    params = config["model"].copy()
    params["scale_pos_weight"] = scale_pos_weight
    params["n_estimators"]     = config["training"]["n_estimators"]
    params["random_state"]     = config["training"]["random_state"]

    N_FOLDS = config["training"]["n_folds"]

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=config["training"]["random_state"],
    )

    oof_preds  = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    scores     = []
    feature_imp = pd.DataFrame()

    # LightGBM callbacks
    callbacks = []
    if hasattr(lgb, "early_stopping"):
        callbacks.append(
            lgb.early_stopping(
                config["training"]["early_stopping_rounds"], verbose=False
            )
        )
    if hasattr(lgb, "log_evaluation"):
        callbacks.append(lgb.log_evaluation(config["training"]["log_freq"]))

    logger.info(f"Training LightGBM with {N_FOLDS}-fold CV …")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        Xtr,  Xval  = X.iloc[tr_idx],  X.iloc[val_idx]
        ytr,  yval  = y.iloc[tr_idx],  y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)], callbacks=callbacks)

        oof_preds[val_idx]  = model.predict_proba(Xval)[:, 1]
        test_preds         += model.predict_proba(X_test)[:, 1] / N_FOLDS

        auc = roc_auc_score(yval, oof_preds[val_idx])
        scores.append(auc)

        logger.info(f"Fold {fold + 1:2d}/{N_FOLDS}  AUC: {auc:.5f}")
        mlflow.log_metric(f"fold_{fold + 1}_auc", auc)

        imp = pd.DataFrame({
            "feature":    feature_names,
            "importance": model.feature_importances_,
            "fold":       fold + 1,
        })
        feature_imp = pd.concat([feature_imp, imp], axis=0, ignore_index=True)

    # Overall OOF AUC
    oof_auc = roc_auc_score(y, oof_preds)

    mlflow.log_metric("oof_auc",       oof_auc)
    mlflow.log_metric("mean_fold_auc", float(np.mean(scores)))
    mlflow.log_metric("std_fold_auc",  float(np.std(scores)))

    logger.info(f"OOF AUC:       {oof_auc:.5f}")
    logger.info(
        f"Mean fold AUC: {np.mean(scores):.5f} ± {np.std(scores):.5f}"
    )

    return {
        "model":              model,   # last-fold model — used for calibration
        "oof_preds":          oof_preds,
        "test_preds":         test_preds,
        "scores":             scores,
        "oof_auc":            oof_auc,
        "feature_importance": feature_imp,
    }


# -------------------------------------------------
# 🔹 Calibration
# -------------------------------------------------

def calibrate_model(
    model: lgb.LGBMClassifier,
    X: np.ndarray,
    y: np.ndarray,
    config: dict,
) -> CalibratedClassifierCV:
    """
    Wraps the last-fold model with Platt scaling (sigmoid) calibration.

    Note: calibration is fitted on the full training set.  The resulting
    calibrated probabilities are used for threshold optimisation and final
    evaluation.  The ensemble test_preds (averaged across folds) are
    intentionally replaced here because the calibrated model has access
    to full training data signal.
    """
    logger.info(
        f"Calibrating probabilities "
        f"(method={config['calibration']['method']}, "
        f"cv={config['calibration']['cv']}) …"
    )
    calibrator = CalibratedClassifierCV(
        model,
        method=config["calibration"]["method"],
        cv=config["calibration"]["cv"],
    )
    calibrator.fit(X, y)
    return calibrator


# -------------------------------------------------
# 🔹 Prediction
# -------------------------------------------------

def predict(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]