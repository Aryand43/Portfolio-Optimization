"""
ml_pipeline.py

Purpose:
- Provide machine learning capabilities for predicting future returns based on historical data.
- Incorporate time series modeling techniques to forecast and enhance portfolio optimization.

Key Functions:
1. prepare_features(data, lag=5):
   - Generates lagged features from historical returns data for model training.

2. train_ml_model(features, target):
   - Trains a simple linear regression model to predict future returns.
   - Splits the dataset into training and test sets and returns both the trained model and predictions.

3. predict_future(model, latest_data, days=5):
   - Uses the trained model to predict returns for the specified number of future days.

Enhancements:
- Implemented feature engineering to create lagged returns and other relevant time series features.
- Added error handling to ensure sufficient data before running the model.
- Modular approach allows for replacing linear regression with more advanced ML models in the future.

Usage:
- These functions provide the basis for predictive analytics within a portfolio optimization framework, allowing for data-driven decision making.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def prepare_features(data, lag=5):
    """
    Create lagged features for time series data.

    Args:
        data (pd.Series): Time series data (e.g., daily returns).
        lag (int): Number of lagged days to include.

    Returns:
        pd.DataFrame: Features and target for training ML models.
    """
    if len(data) <= lag:
        raise ValueError(f"Not enough data to create lagged features with lag={lag}. Provide more data or reduce lag.")
    
    df = pd.DataFrame(data, columns=['returns'])
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = df['returns'].shift(i)
    df.dropna(inplace=True)
    
    if df.empty:
        raise ValueError("Lagging resulted in an empty dataset. Please select a larger date range.")
    
    return df.iloc[:, 1:], df.iloc[:, 0]

def train_ml_model(features, target):
    """
    Train a simple linear regression model on lagged features.

    Args:
        features (pd.DataFrame): Lagged features.
        target (pd.Series): Target values.

    Returns:
        model (LinearRegression): Trained model.
        predictions (np.array): Predictions on test data.
    """
    if len(features) == 0 or len(target) == 0:
        raise ValueError("Not enough data to train the model. Please provide more historical data.")
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model Mean Squared Error: {mse:.4f}")
    return model, predictions

def predict_future(model, latest_data, days=5):
    """
    Predict future returns using the trained model.

    Args:
        model (LinearRegression): Trained ML model.
        latest_data (pd.DataFrame): Latest lagged data for predictions.
        days (int): Number of future days to predict.

    Returns:
        np.array: Predicted future returns.
    """
    predictions = []
    for _ in range(days):
        if latest_data.shape[1] < model.coef_.shape[0]:
            raise ValueError("Insufficient lagged data for prediction. Ensure enough historical data is provided.")
        
        pred = model.predict(latest_data[-1:].values)
        predictions.append(pred[0])
        latest_data = pd.DataFrame(np.append(latest_data.iloc[0, 1:].values, pred[0]).reshape(1, -1))
    return predictions
