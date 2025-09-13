# steps/data_ingestion_step.py
import pandas as pd
from zenml import step
import logging
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingest_data import DataIngestorFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """
    ZenML step that ingests data from file_path.
    It auto-detects .zip/.csv and uses the appropriate ingestor.
    """
    if not file_path:
        raise ValueError("file_path must be provided to data_ingestion_step")

    file_extension = os.path.splitext(file_path)[1].lower()
    logger.info("Detected file extension: %s", file_extension)

    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    if data_ingestor is None:
        raise RuntimeError(f"No data ingestor available for extension: {file_extension}")

    # Use the standardized method name ingest_data
    df = data_ingestor.ingest_data(file_path)
    logger.info("Ingested dataframe shape: %s", getattr(df, "shape", None))
    return df


if __name__ == "__main__":
    # quick smoke test (adjust path as needed)
    test_path = r"C:\Users\ayush\OneDrive\Desktop\new model\data\raw\house_price.zip"
    df = data_ingestion_step(test_path)
    print(df.head())
