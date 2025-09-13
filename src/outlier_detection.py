import logging
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        
        pass
    
class ZScoreOutlierDetectionStrategy(OutlierDetectionStrategy):
    def __init__(self,threshold=3):
        self.threshold=threshold
        
    def detect_outliers(self, df:pd.DataFrame)->pd.DataFrame:
        logging.info("detecting outliers using the z-score method")
        z_scores=np.abs((df-df.mean())/df.std())
        outliers=z_scores > self.threshold
        logging.info(f"outlier detected with z-score thereshold :{self.detect_outliers}")
        return outliers
    
class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df:pd.DataFrame)-> pd.DataFrame:
        logging.info("Detecting outliers using the iqr method")
        Q1=df.quantile(0.25)
        Q3=df.quantile(0.75)
        IQR=Q3-Q1
        outliers=(df<(Q1-1.5*IQR))| (df>(Q3+1.5*IQR))
        logging.info("outlier detected using the iqr method")
        return outliers
    
class OutlierDetector:
    def __init__(self,strategy:OutlierDetectionStrategy):
        self._strategy=strategy
        
    def set_strategy(self,strategy:OutlierDetectionStrategy):
        logging.info("switching outlier detection strategy")
        self._strategy=strategy
        
    def detect_outliers(self,df:pd.DataFrame)->pd.DataFrame:
        logging.info("execting outlier detection strategy")
        return self._strategy.detect_outliers(df)
    
    def handle_outliers(self,df:pd.DataFrame,method="remove")->pd.DataFrame:
        outliers=self.detect_outliers(df)
        if method=="remove":
            logging.info("removing outliers from the dataframe")
            df_cleaned=df[(~outliers).all(axis=1)]
        
        elif method == "cap":
            logging.info("removing outliers from the data set")
            df_cleaned=df.clip(lower=df.quantile(0.01),upper=df.quartile(0.99),axis=1)
        else:
            logging.warning(f"unknown method '{method}', no utliers") 
            return df
        logging.info("outlier handling completed")
        return df_cleaned
    
    def visualize_outliers(self,df:pd.DataFrame,features:list):
        logging.info(f"visualizing outliers for features :{features}")
        for feature in features:
            plt.figure(figsize=(10,6))
            sns.boxplot(x=df[feature])
            plt.title(f"boxplot of feature {feature}")
            plt.show()
        logging.info("outlier visuatization complete")
        
if __name__=="__main__":
    df=pd.read_csv(r"C:\Users\ayush\OneDrive\Desktop\new model\data\raw\house_price.csv")
    df_numeric=df.select_dtypes(include=[np.number]).dropna()
    outlier_detector = OutlierDetector(ZScoreOutlierDetectionStrategy(threshold=3))
    
    outliers=outlier_detector.detect_outliers(df_numeric)
    df_cleaned=outlier_detector.handle_outliers(df_numeric,method="remove")
    print(df_cleaned.shape)
    outlier_detector.visualize_outliers(df_cleaned,features=["Sale Price","Flat Area (in Sqft)"])
    pass
  
    
        