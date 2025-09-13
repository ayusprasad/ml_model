# steps/model_building_step.py
import logging
import sys
import os
from typing import List

import pandas as pd
import numpy as np
import mlflow
from zenml import step
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline as SklearnPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# optionally set a local mlflow tracking uri (uncomment & adjust if needed)
# mlflow.set_tracking_uri("file:///C:/Users/ayush/OneDrive/Desktop/new model/mlruns")


@step(enable_cache=False)
def model_building_step(X_train: pd.DataFrame, y_train: pd.Series) -> SklearnPipeline:
    # basic checks
    if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
        raise ValueError("X_train must be a pandas DataFrame and y_train must be a pandas Series.")

    # infer categorical columns: object/category OR numeric with small number of unique values
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    # treat numeric columns with low cardinality as categorical (tweak threshold as needed)
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_but_cat = [c for c in numeric_cols if X_train[c].nunique() <= 30]  # threshold 30
    # remove columns that are the target or index if present (defensive)
    # ensure we don't double-add numeric_but_cat if they are already object
    for c in numeric_but_cat:
        if c not in categorical_cols:
            categorical_cols.append(c)

    # final numerical cols = numeric columns that are NOT used as categorical
    numerical_cols = [c for c in numeric_cols if c not in categorical_cols]

    logger.info("Categorical columns: %s", categorical_cols)
    logger.info("Numerical columns: %s", numerical_cols)

    # Define transformers
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

    # Ensure MLflow experiment exists and set it
    EXPERIMENT_NAME = "price_prediction_experiment"
    try:
        # This will create the experiment if it doesn't exist
        mlflow.set_experiment(EXPERIMENT_NAME)
    except Exception as e:
        logger.warning("Could not set/create MLflow experiment: %s", e)

    # Enable autologging
    try:
        mlflow.sklearn.autolog()
    except Exception as e:
        logger.warning("mlflow autolog failed: %s", e)

    started_run = False
    try:
        # start run (will now have an experiment due to set_experiment)
        if not mlflow.active_run():
            mlflow.start_run()
            started_run = True

        logger.info("Fitting pipeline...")
        pipeline.fit(X_train, y_train)
        logger.info("Pipeline fit complete")

        # Try to extract expected columns after preprocessing
        expected_columns = list(numerical_cols)
        try:
            ct = pipeline.named_steps["preprocessor"]
            # after fit, named_transformers_ exists
            named_tr = getattr(ct, "named_transformers_", {})
            cat_pipe = named_tr.get("cat", None)
            if cat_pipe is not None and hasattr(cat_pipe, "named_steps"):
                # pipeline with steps -> get onehot step
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

    return pipeline


if __name__ == "__main__":
    print("Sanity check: model_building_step loaded")
