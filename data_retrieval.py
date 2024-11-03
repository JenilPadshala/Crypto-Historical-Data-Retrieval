import os
import time
from dotenv import load_dotenv
import requests
import pandas as pd
from datetime import datetime
import utils

load_dotenv()  # Loads variables from .env file
API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")


def fetch_crypto_data(crypto_pair: str, start_date: str) -> pd.DataFrame | None:
    """
    Fetch historical cryptocurrency data for a specified crypto pair starting from a given date.

    Args:
        crypto_pair (str): The cryptocurrency pair to fetch data for (e.g., 'BTC/USD').
        start_date (str): The start date for fetching data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame | None: A DataFrame containing the historical data, or None if no data is found.
    """
    crypto_pair = crypto_pair.replace("/", "-")
    start_timestamp = utils.date_str_to_unix_timestamp(start_date)
    all_data = pd.DataFrame()
    to_ts = int(datetime.now().timestamp())

    while to_ts > start_timestamp:
        try:
            response = requests.get(
                API_URL,
                params={
                    "market": "cadli",
                    "instrument": crypto_pair,
                    "limit": 5000,
                    "aggregate": 1,
                    "fill": "true",
                    "apply_mapping": "true",
                    "response_format": "JSON",
                    "to_ts": to_ts,
                    "api_key": API_KEY,
                },
                headers={"Content-type": "application/json; charset=UTF-8"},
            )

            # Check if the response is successful
            response.raise_for_status()

            data = response.json().get("Data", [])
            batch_data = utils.parse_response_data(data)

            if batch_data.empty:
                print("No more data available.")
                break

            all_data = pd.concat([all_data, batch_data])

            # update to_ts to the minimum date in the batch
            to_ts = utils.date_str_to_unix_timestamp(
                batch_data["Date"].min().strftime("%Y-%m-%d")
            )
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"API Request failed: {e}")
            return None

    return utils.filter_data_by_date(all_data, start_date)
