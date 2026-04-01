"""
utilities.py
-----------
Utility functions for data loading, preprocessing, and evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Tuple, Dict


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load housing dataset from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    df = pd.read_csv(filepath)
    return df


def explore_dataset(df: pd.DataFrame) -> None:
    """
    Display basic information about the dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
    """
    print("Dataset Overview:")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumn Names:\n{df.columns.tolist()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")


def preprocess_data(df: pd.DataFrame, columns_to_drop: list = None) -> pd.DataFrame:
    """
    Preprocess the dataset by removing unnecessary columns.
    
    Args:
        df (pd.DataFrame): Input dataset
        columns_to_drop (list): List of column names to remove
        
    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    df_processed = df.copy()
    
    if columns_to_drop:
        df_processed.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    return df_processed


def get_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get descriptive statistics of the dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Descriptive statistics
    """
    return df.describe()


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix for all numeric features.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    return df.corr()


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate regression model using multiple metrics.
    
    Args:
        y_true (np.ndarray): Actual values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics
            - 'r2': R² score
            - 'mae': Mean Absolute Error
            - 'mse': Mean Squared Error
            - 'rmse': Root Mean Squared Error
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'r2': r2,
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }


def print_evaluation_metrics(metrics: Dict[str, float]) -> None:
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics (Dict[str, float]): Dictionary of metrics from evaluate_model()
    """
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"R² Score:                  {metrics['r2']:.4f}")
    print(f"Mean Absolute Error (MAE):  ${metrics['mae']:,.2f}")
    print(f"Mean Squared Error (MSE):   ${metrics['mse']:,.2f}")
    print(f"Root Mean Squared Error:    ${metrics['rmse']:,.2f}")
    print("="*50)


def get_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate residuals (errors) between actual and predicted values.
    
    Args:
        y_true (np.ndarray): Actual values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        np.ndarray: Residuals
    """
    return y_true.values - y_pred if isinstance(y_true, pd.Series) else y_true - y_pred
