from abc import ABC, abstractmethod
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, column: str):
        pass
    
    
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df:pd.DataFrame, feature:str):
        plt.figure(figsize=(10,6))
        sns.histplot(df[feature], kde=True)
        plt.title(f'Histogram and KDE of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.show()
        logging.info(f"analyzed numerical feature: {feature}")
        
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df:pd.DataFrame, feature:str):
        plt.figure(figsize=(10,6))
        sns.countplot(data=df,x=feature)
        plt.title(f'Count Plot of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()
        logging.info(f"analyzed categorical feature: {feature}")

class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy
        
    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        logging.info("switching univariate analysis strategy")
        self._strategy = strategy
        
    def execute_analysis(self, df: pd.DataFrame, feature: str):
        logging.info("executing univariate analysis strategy")
        self._strategy.analyze(df, feature)
        
if __name__ == "__main__":
    df=pd.read_csv(r"C:\Users\ayush\OneDrive\Desktop\new model\data\raw\house_price.csv")
    analyzer=UnivariateAnalyzer(NumericalUnivariateAnalysis())
    analyzer.execute_analysis(df, feature="Flat Area (in Sqft)")
    analyzer.set_strategy(CategoricalUnivariateAnalysis())
    analyzer.execute_analysis(df, feature="No of Bathrooms")
    pass