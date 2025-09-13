from abc import ABC, abstractmethod
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """Template method for analyzing missing values"""
        self.identify_missing_values(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        pass


class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        logging.info("Identifying missing values...")

        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]

        if missing_counts.empty:
            print("No missing values found ✅")
        else:
            print("Missing Values Count by Column:")
            print(missing_counts)

        return missing_counts

    def visualize_missing_values(self, df: pd.DataFrame):
        print("\nVisualizing Missing Values...")

        # Heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()

        # Barplot (only if there are missing values)
        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]

        if not missing_counts.empty:
            plt.figure(figsize=(12, 6))
            sns.barplot(x=missing_counts.index, y=missing_counts.values, color="skyblue")
            plt.xticks(rotation=45)
            plt.title("Missing Values Count per Column")
            plt.ylabel("Count")
            plt.show()


if __name__ == "__main__":
    # Example run
    df = pd.read_csv(r"C:\Users\ayush\OneDrive\Desktop\new model\data\raw\house_price.csv")

    missing_values_analyzer = SimpleMissingValuesAnalysis()
    missing_values_analyzer.analyze(df)
