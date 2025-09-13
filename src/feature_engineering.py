import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 


class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self,df:pd.DataFrame):
        pass
    
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self,features):
        self.features=features
    
    def apply_transformation(self,df:pd.DataFrame):
        logging.info("Applying Log Transformation")
        df_transformed=df.copy()
        for feature in self.features:
            df_transformed[feature]=np.log1p(df[feature])
        logging.info("Completed Log Transformation")
        return df_transformed

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,features):
        self.features=features
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        logging.info("Applying Log Transformation")
        X_transformed=X.copy()
        for feature in self.features:
            if (X[feature]<0).any():
                logging.warning(f"{feature} has values<-1 . log1p may return NaN for these values")
            X_transformed[feature]=np.log1p(X[feature])
        return X_transformed

class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self,features):
        self.features=features
        self.scaler=StandardScaler()
    
    def apply_transformation(self, df:pd.DataFrame):
        logging.info("Applying Standard Scaling")
        df_transformed=df.copy()
        df_transformed[self.features]=self.scaler.fit_transform(df[self.features])
        return df_transformed

class StandardScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,features):
        self.features=features
        self.scaler=StandardScaler()
    
    def fit(self,X,y=None):
        self.scaler.fit(X[self.features])
        return self
    
    def transform(self,X):
        logging.info("Applying Standard Scaling")
        X_transformed=X.copy()
        X_transformed[self.features]=self.scaler.transform(X[self.features])
        return X_transformed


class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self,features,feature_range=(0,1)):
        self.features=features
        self.scaler=MinMaxScaler(feature_range=feature_range)
        
    def apply_transformation(self, df):
        logging.info("Applying Min-Max Scaling")
        df_transformed=df.copy()
        df_transformed[self.features]=self.scaler.fit_transform(df[self.features])
        return df_transformed
    
class MinMaxScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,features,feature_range=(0,1)):
        self.features=features
        self.scaler=MinMaxScaler(feature_range=feature_range)
        
    def fit(self,X,y=None):
        self.scaler.fit(X[self.features])
        return self
    
    def transform(self,X):
        logging.info("Applying Min-Max Scaling")
        X_transformed=X.copy()
        X_transformed[self.features]=self.scaler.transform(X[self.features])
        return X_transformed
    
class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self,features):
        self.features=features
        self.encoder=OneHotEncoder(sparse_output=False,drop="first",handle_unknown="ignore")
    
    def apply_transformation(self,df:pd.DataFrame):
        logging.info("Applying one-hot encoding")
        df_transformed=df.copy()
        encoded=self.encoder.fit_transform(df[self.features])
        encoded_df=pd.DataFrame(encoded,columns=self.encoder.get_feature_names_out(self.features),index=df.index)
        df_transformed=df_transformed.drop(columns=self.features)
        df_transformed=pd.concat([df_transformed,encoded_df],axis=1)
        logging.info("Completed one-hot encoding")
        return df_transformed
    
class OneHotEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,features):
        self.features=features
        self.encoder=OneHotEncoder(sparse=False,drop="first",handle_unknown="ignore")
    
    def fit(self,X,y=None):
        self.encoder.fit(X[self.features])
        return self
    
    def transform(self,X):
        logging.info("Applying one-hot encoding")
        X_transformed=X.copy()
        encoded=self.encoder.transform(X[self.features])
        encoded_df=pd.DataFrame(encoded,columns=self.encoder.get_feature_names_out(self.features),index=X.index)
        X_transformed=X_transformed.drop(columns=self.features)
        X_transformed=pd.concat([X_transformed,encoded_df],axis=1)
        return X_transformed
    
class FeatureEngineer:
    def __init__(self,strategy:FeatureEngineeringStrategy):
        self.strategy=strategy
    
    def set_strategy(self,strategy:FeatureEngineeringStrategy):
        logging.info("Changing feature engineering strategy")
        self.strategy=strategy
    
    def apply_feature_engineering(self,df:pd.DataFrame):
        logging.info("Transforming data using the current strategy")
        return self.strategy.apply_transformation(df)

if __name__=="__main__":
    df=pd.read_csv(r"data\raw\house_price.csv")
    
    log_transformer=FeatureEngineer(LogTransformation(features=["Sale Price"]))
    df_log_transformed=log_transformer.apply_feature_engineering(df)
    
    standard_scaler=FeatureEngineer(StandardScaling(features=["Latitude", "Longitude"]))
    df_standard_scaled=standard_scaler.apply_feature_engineering(df)
    
    minmax_scaler=FeatureEngineer(MinMaxScaling(features=["No of Bedrooms","No of Floors"]))
    df_minmax_scaled=minmax_scaler.apply_feature_engineering(df)
    
    onehot_encoder=FeatureEngineer(OneHotEncoding(features=["Waterfront View","Zipcode"]))
    df_onehot_encoded=onehot_encoder.apply_feature_engineering(df)
    
    pass