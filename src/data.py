# src/data.py

import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from features import (
    application_features,
    build_bureau_features,
    build_bureau_balance_features,
)
from utils import reduce_mem

logger = logging.getLogger("ml_pipeline")


# -------------------------------------------------
# 🔹 Load & Feature Engineering
# -------------------------------------------------

def load_and_prepare_data(config: dict) -> pd.DataFrame:
    """
    Loads the main application CSV, runs feature engineering, merges
    bureau and bureau_balance aggregations, label-encodes categoricals,
    aligns columns, and reduces memory usage.
    """
    logger.info("Loading main application data …")
    df = pd.read_csv(config["data"]["main"])

    # --- Feature engineering ---
    logger.info("Running application_features …")
    df = application_features(df)

    logger.info("Building bureau features …")
    bureau = build_bureau_features(config["data"]["bureau"])

    logger.info("Building bureau_balance features …")
    bb = build_bureau_balance_features(
        config["data"]["bb"],
        config["data"]["bureau"],
    )

    # --- Merges ---
    df = df.merge(bureau, on="SK_ID_CURR", how="left")
    df = df.merge(bb, on="SK_ID_CURR", how="left")

    logger.info(f"Shape after merges: {df.shape}")

    # --- Label encode all object columns BEFORE fillna(0) ---
    # fillna(0) on string columns would create a literal "0" string,
    # which corrupts the encoding. Encode first, then fill numeric NaNs.
    logger.info("Label-encoding categorical columns …")
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("Unknown")
        le.fit(df[col])
        df[col] = le.transform(df[col])

    # --- Fill remaining (numeric) NaNs ---
    df.fillna(0, inplace=True)

    # --- Reduce memory ---
    df = reduce_mem(df)

    return df


# -------------------------------------------------
# 🔹 Train / Test Split
# -------------------------------------------------

def split_data(df: pd.DataFrame, config: dict):
    train, test = train_test_split(
        df,
        test_size=0.2,
        stratify=df["TARGET"],
        random_state=config["training"]["random_state"],   # use config, not hardcoded 31
    )
    logger.info(f"Train: {train.shape}  |  Test: {test.shape}")
    return train, test


# -------------------------------------------------
# 🔹 Prepare X / y Arrays
# -------------------------------------------------

def prepare_xy(train: pd.DataFrame, test: pd.DataFrame):
    """
    Aligns train and test columns (inner join) to guard against any
    column mismatch introduced during feature engineering, then returns
    numpy arrays ready for LightGBM.
    """
    y_train = train["TARGET"]
    y_test  = test["TARGET"]

    X_train = train.drop(["TARGET", "SK_ID_CURR"], axis=1)
    X_test  = test.drop(["TARGET", "SK_ID_CURR"], axis=1)

    # Align ensures identical columns in identical order
    X_train, X_test = X_train.align(X_test, join="inner", axis=1)

    logger.info(
        f"Feature matrix — Train: {X_train.shape}  |  Test: {X_test.shape}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        list(X_train.columns),   # feature names returned alongside arrays
    )