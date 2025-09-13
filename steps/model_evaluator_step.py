# steps/model_evaluator_step.py
import logging
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from zenml import step
from src.model_evaluator import ModelEvaluator, RegressionEvaluationStrategy
import pandas as pd
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@step(enable_cache=False)
def model_evaluator_step(
    trained_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> tuple[dict, float]:
    
    if not isinstance(X_test, pd.DataFrame):
        raise ValueError("X_test should be a pandas DataFrame")
    if not isinstance(y_test, pd.Series):
        raise ValueError("y_test should be a pandas Series")

    logger.info("Applying model to test data")

    # Use the pipeline directly (it handles preprocessing internally)
    evaluator = ModelEvaluator(strategy=RegressionEvaluationStrategy())

    # This evaluate method expects (model, X_test, y_test)
    evaluation_metrics = evaluator.evaluate(trained_model, X_test, y_test)

    if not isinstance(evaluation_metrics, dict):
        raise ValueError("evaluation_metrics should be a dictionary")

    # Use the exact key names returned by RegressionEvaluationStrategy
    mse = evaluation_metrics.get("Mean Squared Error", None)
    logger.info(f"Evaluation complete. MSE: {mse}")

    return evaluation_metrics, mse


if __name__ == "__main__":
    print("Testing model evaluator step...")
