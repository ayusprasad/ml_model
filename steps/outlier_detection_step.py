import logging
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from zenml import step
from src.outlier_detection import OutlierDetector,ZScoreOutlierDetectionStrategy


@step
def outlier_detection_step(df:pd.DataFrame,column_name:str) ->pd.DataFrame:
    logging.info(f"starting outlier detection with {column_name} ")

    if df is None:
        logging.error("received a Nonetype Dataframe")
        raise ValueError("input df must be a not null pd df")
    
    if not isinstance(df,pd.DataFrame):
        logging.error(f"Expected pandas dataframe got {type(df)}instead")
        
    
    if column_name not in df.columns:
        logging.error(f"colomn_name {column_name} does not exist")
        raise ValueError(f"column {column_name} does not exist")
    df_numeric=df.select_dtypes(include=[int,float])
    
    outlier_detector=OutlierDetector(ZScoreOutlierDetectionStrategy(threshold=3))
    outliers=outlier_detector.detect_outliers(df_numeric)
    df_cleaned=outlier_detector.handle_outliers(df_numeric,method="remove")

    return df_cleaned

if __name__=="__main__":
       print("outlier detection complete")
    