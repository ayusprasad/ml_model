from abc import ABC, abstractmethod
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        
        pass

class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        logging.info("Inspecting data types")
        print(df.info())
        
class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        logging.info("Inspecting summary statistics")
        print("\n summary statics (numerical features):")
        print(df.describe())
        print("\n summary statics (categorical features):")
        print(df.describe(include=["object"]))
        
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        self._strategy = strategy
        
    def set_strategy(self, strategy: DataInspectionStrategy):
        logging.info("Switching data inspection strategy")
        self._strategy = strategy
        
    def execute_inspection(self, df: pd.DataFrame):
        logging.info("Executing data inspection strategy")
        self._strategy.inspect(df)
        
if __name__ == "__main__":
    # Example usage
    df = pd.read_csv(r"C:\Users\ayush\OneDrive\Desktop\new model\data\raw\house_price.csv")
    
    inspector = DataInspector(DataTypesInspectionStrategy())
    inspector.execute_inspection(df)
    
    inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    inspector.execute_inspection(df)
    pass