import logging
from abc import ABC, abstractmethod
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, data: pd.DataFrame, target_column: str):
        
        pass
    
class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self,test_size=0.3,random_state=34):
        self.test_size=test_size
        self.random_state=random_state
        
    def split_data(self,df:pd.DataFrame,target_column:str):
        logging.info("Starting simple train-test split")
        x=df.drop(columns=[target_column])
        y=df[target_column]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, random_state=self.random_state, shuffle=True,stratify=y if y.nunique() < 20 else None )
        logging.info("Completed simple train-test split")
        logging.info(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")   
        return x_train, x_test, y_train, y_test
        
class DataSplitter:
    def __init__(self,strategy:DataSplittingStrategy):
        self.strategy=strategy
    
    def set_strategy(self,strategy:DataSplittingStrategy):
        logging.info("Changing data splitting strategy")
        self.strategy=strategy
    
    def split(self,df:pd.DataFrame,target_column:str):
        logging.info("Splitting data using the current strategy")
        return self.strategy.split_data(df,target_column)
    
if __name__=="__main__":
    df=pd.read_csv(Path("data/raw/house_price.csv"))
    data_splitter=DataSplitter(SimpleTrainTestSplitStrategy(test_size=0.3,random_state=32))
    x_train, x_test, y_train, y_test=data_splitter.split(df,target_column="Sale Price")
    pass
