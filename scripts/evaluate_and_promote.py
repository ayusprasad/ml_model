# scripts/evaluate_and_promote.py
import joblib
import pandas as pd
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sklearn.metrics import mean_squared_error, r2_score
import shutil
from utils.config_loader import load_config

def evaluate_model(model, X_test, y_test):
    """Evaluates a model and returns a dictionary of metrics."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return {'mse': mse, 'r2': r2}

def main():
    config = load_config()
    
    # Load the test data (you need to save this during your pipeline!)
    test_data = pd.read_csv(config['paths']['test_data'])
    X_test = test_data.drop(columns=[config['data']['target_column']])
    y_test = test_data[config['data']['target_column']]
    
    # Load the challenger model
    challenger_path = config['paths']['challenger_model']
    challenger = joblib.load(challenger_path)
    
    # Load the champion model (if it exists)
    champion_path = config['paths']['champion_model']
    try:
        champion = joblib.load(champion_path)
        champion_metrics = evaluate_model(champion, X_test, y_test)
    except FileNotFoundError:
        print("No champion model found. The challenger will become the first champion.")
        champion_metrics = {'mse': float('inf'), 'r2': -float('inf')} # Set worst possible scores
    
    # Evaluate the challenger
    challenger_metrics = evaluate_model(challenger, X_test, y_test)
    
    print(f"Champion MSE: {champion_metrics['mse']}, R²: {champion_metrics['r2']}")
    print(f"Challenger MSE: {challenger_metrics['mse']}, R²: {challenger_metrics['r2']}")
    
    # Promote if challenger is better (lower MSE)
    if challenger_metrics['mse'] < champion_metrics['mse']:
        print("Challenger is better! Promoting to champion.")
        shutil.copyfile(challenger_path, champion_path)
    else:
        print("Champion remains the best.")

if __name__ == "__main__":
    print("good")
    main()
    print("nice")