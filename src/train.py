# src/train.py

import os

import joblib
import yaml
import mlflow
import mlflow.lightgbm
import mlflow.sklearn

from utils   import get_logger, validate_config
from data    import load_and_prepare_data, split_data, prepare_xy
from model   import train_model, calibrate_model, predict
from evaluate import find_best_threshold, evaluate_model


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    # -------------------------
    # Logging
    # -------------------------
    logger = get_logger("ml_pipeline")
    logger.info("Starting ML pipeline …")

    # -------------------------
    # Config
    # -------------------------
    config = load_config()
    validate_config(config)          # raises KeyError on missing keys

    # -------------------------
    # MLflow run — wraps the entire pipeline
    # -------------------------
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("credit-risk")
    
    with mlflow.start_run(run_name="lgbm_cv"):

        # Log all config sections as MLflow params
        mlflow.log_params({f"model_{k}": v for k, v in config["model"].items()})
        mlflow.log_params({f"training_{k}": v for k, v in config["training"].items()})
        mlflow.log_params({
            "calibration_use":    config["calibration"]["use"],
            "calibration_method": config["calibration"]["method"],
            "calibration_cv":     config["calibration"]["cv"],
            "threshold_min":      config["threshold"]["min"],
            "threshold_max":      config["threshold"]["max"],
            "threshold_steps":    config["threshold"]["steps"],
            "business_loan_amt":  config["business"]["loan_amt"],
            "business_profit_amt": config["business"]["profit_amt"],
        })

        # -------------------------
        # Data
        # -------------------------
        df = load_and_prepare_data(config)
        train, test = split_data(df, config)

        # prepare_xy now also returns feature_names and applies align()
        X_train, X_test, y_train, y_test, feature_names = prepare_xy(train, test)

        # -------------------------
        # Model — K-Fold CV
        # -------------------------
        result = train_model(
            X_train, y_train, X_test, feature_names, config
        )

        model      = result["model"]
        probs_test = result["test_preds"]

        # Save feature importance as a CSV artifact
        fi_path = "feature_importance.csv"
        result["feature_importance"].to_csv(fi_path, index=False)
        mlflow.log_artifact(fi_path)
        logger.info(f"Feature importance saved → {fi_path}")

        # -------------------------
        # Calibration (optional)
        # -------------------------
        if config["calibration"]["use"]:
            model      = calibrate_model(model, X_train, y_train, config)
            probs_test = predict(model, X_test)

        # -------------------------
        # Threshold optimisation
        # -------------------------
        best_t, best_loss = find_best_threshold(y_test, probs_test, config)

        mlflow.log_metric("best_threshold",    best_t)
        mlflow.log_metric("best_business_loss", best_loss)


        # Create artifacts directory
        os.makedirs("artifacts", exist_ok=True)

        # Save model locally
        model_path = "artifacts/model.pkl"
        joblib.dump(model, model_path)
        
        # -------------------------
        # Log the model artifact
        # -------------------------
        if config["calibration"]["use"]:
            # Log calibrated model (sklearn flavor)
            mlflow.sklearn.log_model(model, name="model")
        else:
            # Log LightGBM model
            mlflow.lightgbm.log_model(model, name="model")

        # Log model as artifact (visible in UI)
        mlflow.log_artifact(model_path)

        logger.info(f"Model logged to MLflow (run_id={mlflow.active_run().info.run_id})")

        # -------------------------
        # Evaluation
        # -------------------------
        evaluate_model(y_test, probs_test, best_t)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()