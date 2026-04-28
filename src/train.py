# src/train.py

import os
import json
import joblib
import yaml
import pandas as pd

import mlflow
import mlflow.lightgbm
import mlflow.sklearn

from utils import get_logger, validate_config
from model import train_model, calibrate_model


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    # -------------------------
    # Logging
    # -------------------------
    logger = get_logger("ml_pipeline")
    logger.info("Starting training stage …")

    # -------------------------
    # Config
    # -------------------------
    config = load_config()
    validate_config(config)

    # -------------------------
    # Load TRAIN data (from DVC pipeline)
    # -------------------------
    train = pd.read_csv("data/train.csv")

    y_train = train["TARGET"]
    X_train = train.drop(["TARGET", "SK_ID_CURR"], axis=1)

    feature_names = list(X_train.columns)

    logger.info(f"Training data: {X_train.shape}")

    # -------------------------
    # MLflow setup
    # -------------------------
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("credit-risk")

    with mlflow.start_run(run_name="lgbm_train"):

        # Log params
        mlflow.log_params({f"model_{k}": v for k, v in config["model"].items()})
        mlflow.log_params({f"training_{k}": v for k, v in config["training"].items()})

        # -------------------------
        # Train model
        # -------------------------
        result = train_model(
            X_train,
            y_train,
            None,               # no test here
            feature_names,
            config
        )

        model = result["model"]

        # -------------------------
        # Optional calibration
        # -------------------------
        if config["calibration"]["use"]:
            model = calibrate_model(model, X_train, y_train, config)

        # -------------------------
        # Save artifacts
        # -------------------------
        os.makedirs("artifacts", exist_ok=True)

        model_path = "artifacts/model.pkl"
        joblib.dump(model, model_path)

        # Save feature list (VERY IMPORTANT)
        with open("features.json", "w") as f:
            json.dump(feature_names, f)

        # -------------------------
        # MLflow logging
        # -------------------------
        if config["calibration"]["use"]:
            mlflow.sklearn.log_model(model, name="model")
        else:
            mlflow.lightgbm.log_model(model, name="model")

        mlflow.log_artifact(model_path)
        mlflow.log_artifact("features.json")

        # Feature importance
        if "feature_importance" in result:
            fi_path = "feature_importance.csv"
            result["feature_importance"].to_csv(fi_path, index=False)
            mlflow.log_artifact(fi_path)

        logger.info(f"Model saved → {model_path}")

    logger.info("Training stage complete.")


if __name__ == "__main__":
    main()