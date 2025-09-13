# dynamic_importer.py
import logging
from pathlib import Path
import pandas as pd
from zenml import step

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@step
def dynamic_importer(csv_path: str = "data/raw/house_price.csv", nrows: int | None = None) -> str:
    """
    ZenML step that dynamically imports dataset.
    - csv_path: relative/absolute path to CSV (project root recommended).
    - nrows: load only first n rows (None = all).
    Returns:
      JSON string of DataFrame (orient='split') for portability between steps.
    """
    csv_file = Path(csv_path)
    if csv_file.exists():
        logger.info("Loading dataset from: %s", csv_file)
        df = pd.read_csv(csv_file, nrows=nrows)
    else:
        logger.warning("File %s not found. Falling back to a small inline sample.", csv_file)
        # small sample built from the rows you posted
        sample = {
            "ID": [7129300520, 6414100192, 5631500400],
            "Date House was Sold": ["14-Oct-17", "14-Dec-17", "15-Feb-16"],
            "Sale Price": [221900, 538000, 180000],
            "No of Bedrooms": [3, 3, 2],
            "No of Bathrooms": [1, 2.25, 1],
            "Flat Area (in Sqft)": [1180, 2570, 770],
            "Lot Area (in Sqft)": [5650, 7242, 10000],
            "No of Floors": [1, 2, 1],
            "Waterfront View": ["No", "No", "No"],
            "No of Times Visited": ["None", "None", "None"],
            "Condition of the House": ["Fair", "Fair", "Fair"],
            "Overall Grade": [7, 7, 6],
            "Area of the House from Basement (in Sqft)": [1180, 2170, 770],
            "Basement Area (in Sqft)": [0, 400, 0],
            "Age of House (in Years)": [63, 67, 85],
            "Renovated Year": [0, 1991, 0],
            "Zipcode": [98178, 98125, 98028],
            "Latitude": [47.5112, 47.7210, 47.7379],
            "Longitude": [-122.257, -122.319, -122.233],
            "Living Area after Renovation (in Sqft)": [1340, 1690, 2720],
            "Lot Area after Renovation (in Sqft)": [5650, 7639, 8062],
        }
        df = pd.DataFrame(sample)
        if nrows:
            df = df.head(nrows)

    # Basic cleaning + typing
    #  - trim column names
    df.columns = [c.strip() for c in df.columns]

    # parse date column if present
    date_col = "Date House was Sold"
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False)

    # convert numeric-like columns to numeric (coerce errors)
    # (optional — helps with later pipeline steps)
    for col in df.columns:
        if df[col].dtype == object and col not in (date_col,):
            # try converting
            df[col] = pd.to_numeric(df[col], errors="ignore")

    # Return JSON so ZenML step outputs are simple to serialize
    json_data = df.to_json(orient="split", date_format="iso")
    logger.info("Loaded DataFrame with shape: %s", df.shape)
    return json_data

if __name__ == "__main__":
    print("Testing dynamic importer step...")