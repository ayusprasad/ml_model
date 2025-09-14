# steps/model_building_step.py
import logging
import sys
import os
from typing import List

import pandas as pd
import numpy as np
import mlflow
import joblib  # <-- NEW IMPORT
from zenml import step
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline as SklearnPipeline

# NEW: Import the config loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.config_loader import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = load_config() # <-- Load config once at the top

@step(enable_cache=False)
def model_building_step(X_train: pd.DataFrame, y_train: pd.Series) -> SklearnPipeline:
    # ... [All your existing data validation and preparation code remains exactly the same] ...
    # basic checks
    if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
        raise ValueError("X_train must be a pandas DataFrame and y_train must be a pandas Series.")

    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_but_cat = [c for c in numeric_cols if X_train[c].nunique() <= 30]
    for c in numeric_but_cat:
        if c not in categorical_cols:
            categorical_cols.append(c)
    numerical_cols = [c for c in numeric_cols if c not in categorical_cols]

    logger.info("Categorical columns: %s", categorical_cols)
    logger.info("Numerical columns: %s", numerical_cols)

    numerical_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])

    EXPERIMENT_NAME = "price_prediction_experiment"
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
    except Exception as e:
        logger.warning("Could not set/create MLflow experiment: %s", e)

    try:
        mlflow.sklearn.autolog()
    except Exception as e:
        logger.warning("mlflow autolog failed: %s", e)

    started_run = False
    try:
        if not mlflow.active_run():
            mlflow.start_run()
            started_run = True

        logger.info("Fitting pipeline...")
        pipeline.fit(X_train, y_train)
        logger.info("Pipeline fit complete")

        # ************** NEW CODE STARTS HERE **************
        # Save the trained model to the filesystem as a challenger
        # Use the path from our config file
        model_path = config['paths']['challenger_model']
        
        # Crucial: Create the directory if it doesn't exist
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True) 
        
        # Save the entire pipeline (preprocessor + model) using joblib
        joblib.dump(pipeline, model_path)
        logger.info("Model successfully saved to: %s", model_path)
        # ************** NEW CODE ENDS HERE **************

        # ... [Your existing code for expected_columns remains the same] ...
        expected_columns = list(numerical_cols)
        try:
            ct = pipeline.named_steps["preprocessor"]
            named_tr = getattr(ct, "named_transformers_", {})
            cat_pipe = named_tr.get("cat", None)
            if cat_pipe is not None and hasattr(cat_pipe, "named_steps"):
                onehot = cat_pipe.named_steps.get("onehot", None)
            else:
                onehot = cat_pipe

            if onehot is not None and hasattr(onehot, "get_feature_names_out") and len(categorical_cols) > 0:
                try:
                    ohe_names = onehot.get_feature_names_out(categorical_cols)
                    expected_columns.extend(list(ohe_names))
                except Exception:
                    try:
                        ohe_names = onehot.get_feature_names_out()
                        expected_columns.extend(list(ohe_names))
                    except Exception:
                        logger.debug("Could not get OHE feature names; skipping.")
        except Exception as e:
            logger.debug("Error building expected columns list: %s", e)

        logger.info("Expected feature count (head): %s", expected_columns[:20])

    except Exception as e:
        logger.exception("An error occurred during model building: %s", e)
        raise
    finally:
        if started_run and mlflow.active_run():
            mlflow.end_run()

    return pipeline # Return the pipeline as before


if __name__ == "__main__":
    print("Sanity check: model_building_step loaded")