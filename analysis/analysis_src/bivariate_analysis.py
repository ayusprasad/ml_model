from abc import ABC, abstractmethod
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, x: str, y: str):
        pass
    
class NumericalVsNumericalStrategy(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, x: str, y: str):
        if not (np.issubdtype(df[x].dtype, np.number) and np.issubdtype(df[y].dtype, np.number)):
            raise ValueError("Both x and y must be numerical columns")
        
        logging.info(f"Analyzing numerical vs numerical: {x} vs {y}")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=x, y=y)
        plt.title(f'Scatter Plot of {y} vs {x}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

class CategoricalVsNumericalStrategy(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, x: str, y: str):
        if not (df[x].dtype == 'object' and np.issubdtype(df[y].dtype, np.number)):
            raise ValueError("x must be a categorical column and y must be a numerical column")
        
        logging.info(f"Analyzing categorical vs numerical: {x} vs {y}")
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x=x, y=y)
        plt.title(f'Box Plot of {y} by {x}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.xticks(rotation=45)
        plt.show()
        
class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        self._strategy = strategy
        
    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        logging.info("Switching bivariate analysis strategy")
        self._strategy = strategy
        
    def execute_analysis(self, df: pd.DataFrame, x: str, y: str):
        logging.info("Executing bivariate analysis strategy")
        self._strategy.analyze(df, x, y)
        
        
if __name__ == "__main__":
    df=pd.read_csv(r"C:\Users\ayush\OneDrive\Desktop\new model\data\raw\house_price.csv")
    analyzer = BivariateAnalyzer(NumericalVsNumericalStrategy())    
    analyzer.execute_analysis(df, x="Flat Area (in Sqft)", y="Sale Price")
    analyzer.set_strategy(CategoricalVsNumericalStrategy())
    analyzer.execute_analysis(df, x="Condition of the House", y="Sale Price")
    pass