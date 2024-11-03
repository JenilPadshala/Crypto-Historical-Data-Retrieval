from typing import Tuple, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from utils import evaluation_metrics


def train_model(
    data: pd.DataFrame,
    variable1: int,
    variable2: int,
    test_size: float = 0.2,
    random_state: int = 0,
) -> Tuple[Any, float, float, float]:
    """
    Trains an XGBoost model on the input data and returns the trained model along with the evaluation metrics on the test set.

    Parameters:
    data (pd.DataFrame): The input data containing feature and target columns.
    variable1 (int): The number of days to consider for historical metrics.
    variable2 (int): The number of days to consider for future metrics.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): The seed used by the random number generator.

    Returns:
    Tuple[Any, float, float, float]: A tuple containing the trained model and the evaluation metrics (MSE, MAE, R2) on the test set.
    """
    # 1. Extract the feature columns and target columns, and split the dataset into training and testing sets
    feature_columns = [
        f"Days_Since_High_Last_{variable1}_Days",
        f"%_Diff_From_High_Last_{variable1}_Days",
        f"Days_Since_Low_Last_{variable1}_Days",
        f"%_Diff_From_Low_Last_{variable1}_Days",
    ]
    target_columns = [
        f"%_Diff_From_High_Next_{variable2}_Days",
        f"%_Diff_From_Low_Next_{variable2}_Days",
    ]
    X = data[feature_columns]
    y = data[target_columns]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 2. Initialize the XGBoost regressor and wrap it in MultiOutputRegressor
    xgb_model = MultiOutputRegressor(
        XGBRegressor(objective="reg:squarederror", random_state=random_state)
    )
    xgb_model.fit(X_train, y_train)

    # 3. Make predictions on the training set and calculate evaluation metrics
    predictions = xgb_model.predict(X_train)
    mse, mae, r2 = evaluation_metrics(y_train, predictions)

    # 5. Evaluate the model on the test set
    test_predictions = xgb_model.predict(X_test)
    test_mse, test_mae, test_r2 = evaluation_metrics(y_test, test_predictions)

    return xgb_model, test_mse, test_mae, test_r2


def predict_outcomes(model: Any, X: np.ndarray) -> np.ndarray:
    """
    This function takes a trained model and a set of input features and returns the predicted outcomes.

    Parameters:
    model (Any): The trained model.
    X (np.ndarray): The input features.

    Returns:
    np.ndarray: The predicted outcomes.
    """
    return model.predict(X)
