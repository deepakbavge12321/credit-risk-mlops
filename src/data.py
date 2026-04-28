# src/data.py

import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("ml_pipeline")


# -------------------------------------------------
# 🔹 Train / Test Split (DVC Stage)
# -------------------------------------------------

def split_data(df: pd.DataFrame, config: dict):
    train, test = train_test_split(
        df,
        test_size=config["training"]["test_size"],
        stratify=df["TARGET"],
        random_state=config["training"]["random_state"],
    )
    logger.info(f"Train: {train.shape}  |  Test: {test.shape}")
    return train, test


# -------------------------------------------------
# 🔹 Main (DVC Entry Point)
# -------------------------------------------------

def main():
    import yaml

    logger.info("Loading features for split stage …")

    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Load features created by features.py
    df = pd.read_csv("data/features.csv")

    # Split
    train, test = split_data(df, config)

    # Save outputs for next stages
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    logger.info("Saved train.csv and test.csv")


if __name__ == "__main__":
    main()