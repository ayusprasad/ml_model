# src/ingest_data.py
import os
import zipfile
import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataIngestionStrategy(ABC):
    @abstractmethod
    def ingest_data(self, file_path: str) -> pd.DataFrame:
        """Ingest data and return a pandas DataFrame."""
        raise NotImplementedError


class zipDataIngestor(DataIngestionStrategy):
    def ingest_data(self, file_path: str) -> pd.DataFrame:
        file_path = str(file_path)
        if not file_path.lower().endswith(".zip"):
            raise ValueError("Provided file is not a .zip archive: %s" % file_path)

        logging.info("Reading CSV from zip archive: %s", file_path)
        with zipfile.ZipFile(file_path, "r") as zf:
            # find first CSV file in archive
            csv_files = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if not csv_files:
                raise FileNotFoundError("No CSV file found inside the zip archive.")
            if len(csv_files) > 1:
                logging.warning("Multiple CSV files found in the zip archive. Using the first: %s", csv_files[0])

            csv_name = csv_files[0]
            with zf.open(csv_name) as csv_file:
                # read directly from the zip member (binary file-like)
                df = pd.read_csv(csv_file)
        return df

    # keep a compatibility alias in case other code calls ingest(...) 
    def ingest(self, file_path: str) -> pd.DataFrame:
        return self.ingest_data(file_path)


class csvDataIngestor(DataIngestionStrategy):
    def ingest_data(self, file_path: str) -> pd.DataFrame:
        file_path = str(file_path)
        if not file_path.lower().endswith(".csv"):
            raise ValueError("Provided file is not a .csv file: %s" % file_path)

        logging.info("Reading CSV file: %s", file_path)
        df = pd.read_csv(file_path)
        return df

    def ingest(self, file_path: str) -> pd.DataFrame:
        return self.ingest_data(file_path)


class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> Optional[DataIngestionStrategy]:
        file_extension = (file_extension or "").lower()
        if file_extension == ".zip":
            return zipDataIngestor()
        if file_extension == ".csv":
            return csvDataIngestor()

        logging.error("No ingestor available for the file type: %s", file_extension)
        return None


if __name__ == "__main__":
    # quick local test
    file_path = r"C:\Users\ayush\OneDrive\Desktop\new model\data\raw\house_price.zip"
    ext = os.path.splitext(file_path)[1]
    ing = DataIngestorFactory.get_data_ingestor(ext)
    if ing is None:
        raise SystemExit("No ingestor for extension: %s" % ext)
    df = ing.ingest_data(file_path)
    print(df.head())
