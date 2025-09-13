import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        pass


class LinearRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train, y_train):
        if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
            raise TypeError("X_train should be a DataFrame and y_train should be a Series")

        # Store feature information for later access
        self.numeric_features = X_train.select_dtypes(include=[np.number]).columns
        self.categorical_features = X_train.select_dtypes(include=["object"]).columns
        
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features)    
            ])
        
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ])
        
        logging.info("Training Linear Regression model")
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed")
        return pipeline


class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy: ModelBuildingStrategy):
        logging.info("Changing model building strategy")
        self.strategy = strategy
    
    def build_and_train(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        logging.info("Building and training model using the current strategy")
        return self.strategy.build_and_train_model(X_train, y_train)

if __name__ == "__main__":
    df = pd.read_csv("data/raw/house_price.csv")
    X_train = df.drop(columns=["Sale Price"])
    y_train = df["Sale Price"]
    
    strategy = LinearRegressionStrategy()
    model_builder = ModelBuilder(strategy)
    model = model_builder.build_and_train(X_train, y_train)
    
    # Access the feature information from the strategy instance
    numeric_features = strategy.numeric_features
    categorical_features = strategy.categorical_features
    
    preprocessor = model.named_steps['preprocessor']
    feature_names = (numeric_features.tolist() + 
                     preprocessor.named_transformers_["cat"]
                     .named_steps["onehot"]
                     .get_feature_names_out(categorical_features).tolist())
    
    print("Feature names:", feature_names)
    print("Model coefficients:", model.named_steps['model'].coef_)