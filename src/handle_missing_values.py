# src/handle_missing_values.py
import logging
from abc import ABC, abstractmethod
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, axis=0, thresh=None, subset: list | None = None):
        """
        axis: 0 -> drop rows, 1 -> drop cols
        thresh: passed to dropna
        subset: list of columns to consider when dropping rows
        """
        self.axis = axis
        self.thresh = thresh
        self.subset = subset

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Dropping missing values with axis={self.axis}, thresh={self.thresh}, subset={self.subset}")
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh, subset=self.subset)
        logging.info("Missing values dropped")
        return df_cleaned


class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, method="mean", fill_value=None, features: list | None = None):
        """
        method: 'mean', 'median', 'mode', 'constant', 'ffill', 'bfill'
        fill_value: value to use when method == 'constant'
        features: list of columns to apply the filling to (None => sensible defaults)
        """
        self.method = method
        self.fill_value = fill_value
        self.features = features

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Filling missing values using method='{self.method}', features={self.features}, fill_value={self.fill_value}")
        df_cleaned = df.copy()

        # decide which columns to operate on
        if self.features:
            cols = [c for c in self.features if c in df_cleaned.columns]
        else:
            # sensible defaults:
            if self.method in ["mean", "median"]:
                cols = df_cleaned.select_dtypes(include="number").columns.tolist()
            else:
                # for mode / ffill / bfill / constant operate on all columns
                cols = df_cleaned.columns.tolist()

        if self.method == "mean":
            numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df_cleaned[c])]
            if numeric_cols:
                means = df_cleaned[numeric_cols].mean()
                df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(means)
        elif self.method == "median":
            numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df_cleaned[c])]
            if numeric_cols:
                medians = df_cleaned[numeric_cols].median()
                df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(medians)
        elif self.method == "mode":
            for column in cols:
                if not df_cleaned[column].mode().empty:
                    df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].mode()[0])
        elif self.method == "constant":
            df_cleaned[cols] = df_cleaned[cols].fillna(self.fill_value)
        elif self.method in ["ffill", "bfill"]:
            df_cleaned[cols] = df_cleaned[cols].fillna(method=self.method)
        else:
            logging.error(f"Unknown method: {self.method} — no missing values handled")
            return df

        logging.info("Missing values filled")
        return df_cleaned


class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        logging.info("Switching missing value handling strategy")
        self._strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Handling missing values using the selected strategy")
        return self._strategy.handle(df)


if __name__ == "__main__":
    # quick manual smoke test
    df = pd.read_csv(r"C:\Users\ayush\OneDrive\Desktop\new model\data\raw\house_price.csv")
    handler = MissingValueHandler(strategy=DropMissingValuesStrategy(axis=0, thresh=None, subset=None))
    df_cleaned = handler.handle_missing_values(df)
    print("After drop:", df_cleaned.shape)

    handler.set_strategy(FillMissingValuesStrategy(method="mean"))
    df_filled = handler.handle_missing_values(df)
    print("After fill (mean):", df_filled.shape)
