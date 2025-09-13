# steps/feature_engineering_step.py
import logging
import sys
import os
from typing import List

import pandas as pd
import numpy as np
from zenml import step

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_engineering import (
    FeatureEngineer,
    LogTransformation,
    MinMaxScaling,
    OneHotEncoding,
    StandardScaling,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@step(enable_cache=False)
def feature_engineering_step(
    df: pd.DataFrame, strategy: str = "log", features: List[str] | None = None
) -> pd.DataFrame:
    """
    Apply feature engineering transformation specified by `strategy`.
    If `features` is None we infer sensible defaults:
      - numeric strategies ("log", "minmax", "standard") -> numeric cols
      - "onehot" -> object / categorical cols
    """

    logger.info("feature_engineering_step called: strategy=%s features=%s", strategy, features)

    # Validate df
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    # If no features are provided, infer reasonable defaults
    if features is None:
        if strategy in ("log", "minmax", "standard"):
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            logger.info("Inferred numeric features for strategy '%s': %s", strategy, features)
        elif strategy == "onehot":
            features = df.select_dtypes(include=["object", "category"]).columns.tolist()
            logger.info("Inferred categorical features for onehot: %s", features)
        else:
            raise ValueError(f"Unknown strategy '{strategy}' and no features supplied")

    # Keep only valid columns present in DataFrame
    features = [f for f in features if f in df.columns]
    if len(features) == 0:
        raise ValueError("No valid features found in dataframe for the requested strategy")

    # Instantiate the correct strategy object (pass features to strategy constructors)
    if strategy == "log":
        strategy_obj = LogTransformation(features=features)
    elif strategy == "minmax":
        strategy_obj = MinMaxScaling(features=features)
    elif strategy == "onehot":
        strategy_obj = OneHotEncoding(features=features)
    elif strategy == "standard":
        strategy_obj = StandardScaling(features=features)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    # Create FeatureEngineer and run transformation
    feature_engineer = FeatureEngineer(strategy=strategy_obj)
    transformed_df = feature_engineer.apply_feature_engineering(df)

    logger.info("Feature engineering completed. Output shape: %s", transformed_df.shape)
    return transformed_df


if __name__ == "__main__":
    # quick local smoke test
    csv_path = os.path.join("data", "raw", "house_price.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        out = feature_engineering_step(df, strategy="log", features=["Sale Price"])
        print("Returned:", type(out), "shape:", out.shape)
    else:
        print("Local test skipped: no", csv_path)
