# src/utils.py

import logging
import pandas as pd


# -------------------------------------------------
# 🔹 Logging setup
# -------------------------------------------------

def get_logger(name: str = "ml_pipeline") -> logging.Logger:
    """
    Returns a logger that writes to console with a consistent format.
    Call once at the top of train.py; all other modules receive it via
    argument or just use logging.getLogger(__name__).
    """
    logger = logging.getLogger(name)

    if not logger.handlers:                      # avoid duplicate handlers on re-import
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


# -------------------------------------------------
# 🔹 Memory reduction
# -------------------------------------------------

def reduce_mem(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Downcasts numeric columns to the smallest dtype that can represent
    all values without loss.  String / object columns are left untouched
    (they must be encoded before calling this).
    """
    start_mb = df.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df.select_dtypes(include=["int64", "int32", "int16", "int8"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in df.select_dtypes(include=["float64", "float32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    end_mb = df.memory_usage(deep=True).sum() / 1024 ** 2

    if verbose:
        logger = get_logger()
        reduction = 100 * (start_mb - end_mb) / start_mb if start_mb > 0 else 0
        logger.info(
            f"reduce_mem: {start_mb:.1f} MB → {end_mb:.1f} MB "
            f"({reduction:.0f}% reduction)"
        )

    return df


# -------------------------------------------------
# 🔹 Config validation
# -------------------------------------------------

REQUIRED_CONFIG_KEYS = {
    "data":       ["main", "bureau", "bb"],
    "model":      ["objective", "metric", "boosting_type", "learning_rate",
                   "num_leaves", "subsample", "colsample_bytree"],
    "training":   ["n_estimators", "n_folds", "early_stopping_rounds",
                   "log_freq", "random_state"],
    "calibration": ["use", "method", "cv"],
    "threshold":  ["min", "max", "steps"],
    "business":   ["loan_amt", "profit_amt"],
}

def validate_config(config: dict) -> None:
    """
    Raises KeyError with a descriptive message if any required config
    key is missing.  Call this once at startup before any data is loaded.
    """
    for section, keys in REQUIRED_CONFIG_KEYS.items():
        if section not in config:
            raise KeyError(f"Config is missing top-level section: '{section}'")
        for key in keys:
            if key not in config[section]:
                raise KeyError(
                    f"Config section '{section}' is missing required key: '{key}'"
                )