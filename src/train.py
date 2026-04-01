"""
train.py
--------
Main training script for the house price prediction model.

This script orchestrates the full ML pipeline:
1. Data loading and exploration
2. Data preprocessing
3. Feature selection
4. Model training
5. Model evaluation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import utilities
import os


def main():
    """
    Execute the complete machine learning pipeline.
    """
    print("="*60)
    print("HOUSE PRICE PREDICTION - LINEAR REGRESSION")
    print("="*60)
    
    # Configuration
    DATA_PATH = '../data/USA_Housing.csv'
    TEST_SIZE = 0.3
    RANDOM_STATE = 101
    
    FEATURE_COLUMNS = [
        'Avg. Area Income',
        'Avg. Area House Age',
        'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms',
        'Area Population'
    ]
    
    # Step 1: Load Data
    print("\n[Step 1] Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Please ensure the data file exists.")
        return
    
    df = utilities.load_data(DATA_PATH)
    print(f"✓ Data loaded successfully. Shape: {df.shape}")
    
    # Step 2: Explore Dataset
    print("\n[Step 2] Exploring dataset...")
    utilities.explore_dataset(df)
    
    # Step 3: Preprocess Data
    print("\n[Step 3] Preprocessing data...")
    df = utilities.preprocess_data(df, columns_to_drop=['Address'])
    print(f"✓ Data preprocessed. Address column removed.")
    print(f"✓ Statistics:\n{utilities.get_feature_statistics(df).to_string()}")
    
    # Step 4: Analyze Correlations
    print("\n[Step 4] Analyzing feature correlations...")
    correlation_with_price = df.corr()['Price'].sort_values(ascending=False)
    print(f"\nCorrelation with Price:\n{correlation_with_price.to_string()}")
    
    # Step 5: Prepare Features and Target
    print("\n[Step 5] Preparing features and target...")
    X = df[FEATURE_COLUMNS]
    y = df['Price']
    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    
    # Step 6: Split Data
    print("\n[Step 6] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Testing set: {X_test.shape[0]} samples")
    
    # Step 7: Train Model
    print("\n[Step 7] Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("✓ Model training completed.")
    
    # Step 8: Make Predictions
    print("\n[Step 8] Making predictions...")
    y_pred = model.predict(X_test)
    print(f"✓ Predictions generated for {len(y_pred)} test samples.")
    
    # Step 9: Evaluate Model
    print("\n[Step 9] Evaluating model performance...")
    metrics = utilities.evaluate_model(y_test, y_pred)
    utilities.print_evaluation_metrics(metrics)
    
    # Step 10: Display Model Coefficients
    print("\n[Step 10] Model Coefficients:")
    print("-" * 50)
    for feature, coef in zip(FEATURE_COLUMNS, model.coef_):
        print(f"{feature:.<40} {coef:>12,.2f}")
    print(f"{'Intercept':.<40} {model.intercept_:>12,.2f}")
    print("-" * 50)
    
    # Step 11: Analyze Residuals
    print("\n[Step 11] Analyzing residuals...")
    residuals = utilities.get_residuals(y_test, y_pred)
    print(f"✓ Mean residual: ${np.mean(residuals):,.2f}")
    print(f"✓ Std deviation of residuals: ${np.std(residuals):,.2f}")
    print(f"✓ Min residual: ${np.min(residuals):,.2f}")
    print(f"✓ Max residual: ${np.max(residuals):,.2f}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return model, X, y, metrics


if __name__ == '__main__':
    main()
