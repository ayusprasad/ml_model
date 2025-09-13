# steps/handle_missing_values_step.py
import pandas as pd
from zenml import step
import logging
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.handle_missing_values import (
    MissingValueHandler,
    DropMissingValuesStrategy,
    FillMissingValuesStrategy,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# disable cache while you're iterating so old artifacts don't hide bugs
@step(enable_cache=False)
def handle_missing_values_step(
    df: pd.DataFrame,
    strategy: str = "mean",
    features: list | None = None,
    fill_value=None,
) -> pd.DataFrame:
    """
    Handles missing values and returns a cleaned dataframe.

    - strategy: "drop" | "mean" | "median" | "mode" | "constant" | "ffill" | "bfill"
    - features: optional list of columns to apply filling / dropping to
    - fill_value: used when strategy == "constant"
    """
    logger.info("handle_missing_values_step called with strategy=%s features=%s", strategy, features)

    if strategy == "drop":
        handler = MissingValueHandler(strategy=DropMissingValuesStrategy(axis=0, thresh=None, subset=features))
    elif strategy in ["mean", "median", "mode", "constant", "ffill", "bfill"]:
        handler = MissingValueHandler(
            FillMissingValuesStrategy(method=strategy, fill_value=fill_value, features=features)
        )
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    # **IMPORTANT** actually run the handler and return the cleaned dataframe
    df_cleaned = handler.handle_missing_values(df)

    logger.info("handle_missing_values_step completed; cleaned shape=%s", getattr(df_cleaned, "shape", None))
    return df_cleaned


if __name__ == "__main__":
    # Local quick smoke test (run `python steps/handle_missing_values_step.py`)
    import pathlib
    csv_path = pathlib.Path("data/raw/house_price.csv")
    if not csv_path.exists():
        print("Local test skipped: file not found:", csv_path)
    else:
        df = pd.read_csv(csv_path)
        out = handle_missing_values_step(df, strategy="mean", features=None)
        print("Returned type:", type(out), "shape:", getattr(out, "shape", None))
