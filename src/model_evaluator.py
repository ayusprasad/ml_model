import logging
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import RegressorMixin
import numpy as np
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        
        pass
    
class RegressionEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(
        self,model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series)-> dict:
        logging.info("predicting using the trained model")
        y_pred=model.predict(X_test)
        
        logging.info("calculating evaluation metrics")
        mse=mean_squared_error(y_test,y_pred)
        r2=r2_score(y_test,y_pred)
        
        metrics={"Mean Squared Error":mse,"R2 Score":r2}
        logging.info(f"Evaluation metrics: {metrics}")
        return metrics

class ModelEvaluator:
    def __init__(self,strategy:ModelEvaluationStrategy):
        self.strategy=strategy
    
    def set_strategy(self,strategy:ModelEvaluationStrategy):
        logging.info("setting new evaluation strategy")
        
        self.strategy=strategy
    
    def evaluate(self,model:RegressorMixin,X_test:pd.DataFrame,y_test:pd.Series)->dict:
        logging.info("evaluating the model using the selected strategy")
        return self.strategy.evaluate_model(model,X_test,y_test)
    




    
    