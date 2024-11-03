import pandas as pd
from typing import Optional
import utils


def calculate_metrics(
    data: pd.DataFrame, variable1: int, variable2: int
) -> pd.DataFrame:
    """
    Main function to calculate historical and future metrics for stock data.

    This function processes the input DataFrame by converting the 'Date' column to datetime,
    sorting the data by date, and then calling helper functions to add historical and future metrics.

    Args:
        data (pd.DataFrame): The input DataFrame containing stock data with a 'Date' column.
        variable1 (int): The number of days to look back for historical metrics.
        variable2 (int): The number of days to look ahead for future metrics.

    Returns:
        pd.DataFrame: The DataFrame with added historical and future metrics.
    """

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    data = add_historical_metrics(data, variable1)
    data = add_future_metrics(data, variable2)

    return data


def add_historical_metrics(data: pd.DataFrame, variable1: int) -> pd.DataFrame:
    """
    Calculate historical metrics over the past `variable1` days.

    Args:
        data (pd.DataFrame): The input DataFrame containing stock data.
        variable1 (int): The number of days to look back for historical metrics.

    Returns:
        pd.DataFrame: The DataFrame with added historical metrics.
    """
    # Historical high price
    data[f"High_Last_{variable1}_Days"] = utils.calculate_historical_high(
        data, variable1
    )

    # Days since high
    data[f"Days_Since_High_Last_{variable1}_Days"] = [
        utils.days_since_high(data, i, variable1) if i >= variable1 - 1 else 0
        for i in range(len(data))
    ]

    # Percentage difference from historical high
    data[f"%_Diff_From_High_Last_{variable1}_Days"] = utils.calculate_pct_diff(
        data["Close"], data[f"High_Last_{variable1}_Days"]
    )

    # Historical low price
    data[f"Low_Last_{variable1}_Days"] = utils.calculate_historical_low(data, variable1)

    # Days since low
    data[f"Days_Since_Low_Last_{variable1}_Days"] = [
        utils.days_since_low(data, i, variable1) if i >= variable1 - 1 else 0
        for i in range(len(data))
    ]

    # Percentage difference from historical low
    data[f"%_Diff_From_Low_Last_{variable1}_Days"] = utils.calculate_pct_diff(
        data["Close"], data[f"Low_Last_{variable1}_Days"]
    )
    return data


def add_future_metrics(data: pd.DataFrame, variable2: int) -> pd.DataFrame:
    """
    Calculate future metrics over the next `variable2` days.

    Args:
        data (pd.DataFrame): The input DataFrame containing stock data.
        variable2 (int): The number of days to look ahead for future metrics.

    Returns:
        pd.DataFrame: The DataFrame with added future metrics.
    """
    # Future high price
    data[f"High_Next_{variable2}_Days"] = utils.calculate_future_high(data, variable2)

    # Percentage difference from future high
    data[f"%_Diff_From_High_Next_{variable2}_Days"] = utils.calculate_pct_diff(
        data["Close"], data[f"High_Next_{variable2}_Days"]
    )

    # Future low price
    data[f"Low_Next_{variable2}_Days"] = utils.calculate_future_low(data, variable2)

    # Percentage difference from future low
    data[f"%_Diff_From_Low_Next_{variable2}_Days"] = utils.calculate_pct_diff(
        data["Close"], data[f"Low_Next_{variable2}_Days"]
    )
    return data
