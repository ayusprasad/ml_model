# training_pipeline.py
import logging
import sys
import os

# allow running this file from the steps/ folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from steps.data_ingestion_step import data_ingestion_step
from steps.data_splitter_step import data_splitter_step
from steps.feature_engineering_step import feature_engineering_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
from steps.outlier_detection_step import outlier_detection_step
from zenml import Model, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pipeline(
    model=Model(
        name="price_predictor"
    ),
)
def ml_pipeline():
    raw_data = data_ingestion_step(file_path=r"C:\Users\ayush\OneDrive\Desktop\new model\data\raw\house_price.csv")
    
    # pass required args to the missing-values step
    filled_data = handle_missing_values_step(
        raw_data,
        features=["Sale Price", "Area of the House from Basement (in Sqft)"]
    )
    
    engineered_data = feature_engineering_step(
        filled_data, strategy="log", features=["Sale Price", "Area of the House from Basement (in Sqft)"]
    )

    clean_data = outlier_detection_step(engineered_data, column_name="Sale Price")
    X_train, X_test, y_train, y_test = data_splitter_step(clean_data, target_column="Sale Price")

    # IMPORTANT: correct unpacking variable names
    # X_train, X_test, y_train, y_test = data_splitter_step(clean_data, target_column="Sale Price")
    
    model = model_building_step(X_train, y_train=y_train)
    
    evaluation_metrics, mse = model_evaluator_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )
    
    # you may return whatever your pipeline needs; returning the trained model is fine
    return model


if __name__ == "__main__":
    logger.info("Running pipeline (sanity call)")
    # to actually run with ZenML you would use ZenML CLI; calling pipeline() directly just builds it
    pipeline_instance = ml_pipeline()
    print("Built pipeline:", pipeline_instance)
