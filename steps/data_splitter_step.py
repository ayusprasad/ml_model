# steps/data_splitter_step.py
from typing import Tuple
import logging
import pandas as pd
from zenml import step
from src.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy

logging.basicConfig(level=logging.INFO)

@step
def data_splitter_step(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns: X_train, X_test, y_train, y_test
    """
    splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy())
    X_train, X_test, y_train, y_test = splitter.split(df, target_column)
    logging.info("Returning X_train, X_test, y_train, y_test")
    return X_train, X_test, y_train, y_test


# Optional quick local test
if __name__ == "__main__":
    import pathlib
    df = pd.read_csv(pathlib.Path("data/raw/house_price.csv"))
    X_train, X_test, y_train, y_test = data_splitter_step(df, target_column="Sale Price")  # calls step directly
    print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
