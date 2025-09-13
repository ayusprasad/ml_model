from abc import ABC, abstractmethod
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MultivariateAnalysisTemplate(ABC):
    def analyze(self,df:pd.DataFrame):
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)
        
    @abstractmethod
    def generate_correlation_heatmap(self,df:pd.DataFrame):
        pass
    
    @abstractmethod
    def generate_pairplot(self,df:pd.DataFrame):
        pass

class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    def generate_correlation_heatmap(self,df:pd.DataFrame):
        logging.info("generating correlation heatmap")
        plt.figure(figsize=(12,8))
        corr=df.corr()
        sns.heatmap(corr,annot=True,cmap='coolwarm',fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.show()
        
    def generate_pairplot(self,df:pd.DataFrame):
        logging.info("generating pairplot")
        sns.pairplot(df)
        plt.title('Pairplot of Numerical Features')
        plt.show()

if __name__ == "__main__":
    df=pd.read_csv(r"C:\Users\ayush\OneDrive\Desktop\new model\data\raw\house_price.csv")
    multivariate_analyzer=SimpleMultivariateAnalysis()
    selected_features=df[["Flat Area (in Sqft)","Sale Price"]]
    multivariate_analyzer.analyze(selected_features)
    
    pass