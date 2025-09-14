# utils/config_loader.py
import yaml
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_config(config_path="config/config.yaml"):
    """
    Loads and returns the configuration from the YAML file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Example usage in your code later:
# from utils.config_loader import load_config
# config = load_config()
# file_path = config['paths']['raw_data']

if __name__=="__main__":
    print("good")