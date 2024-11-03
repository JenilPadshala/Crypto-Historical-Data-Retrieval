import streamlit as st
import pandas as pd
from data_retrieval import fetch_crypto_data
from metrics_calculations import calculate_metrics
from ml_model import train_model
from utils import add_to_excel

# Streamlit app
st.title("Crypto Historical Data Retrieval")

# Input fields
crypto_pair = st.text_input("Enter Crypto Pair:", "BTC/USD")
start_date = st.date_input("Enter Start Date:", pd.to_datetime("2020-01-01"))

# Dropdowns
variable1 = st.selectbox("Select Variable 1", list(range(1, 101)))
variable2 = st.selectbox("Select Variable 2", list(range(1, 101)))

# Checkbox to ask whether to train a model
train_model_checkbox = st.checkbox("Train Model")


# Button to fetch data
if st.button("Fetch Data"):
    df = fetch_crypto_data(crypto_pair, str(start_date))
    st.success("Data fetched successfully!", icon="✅")
    new_df = calculate_metrics(df, variable1, variable2)
    st.write(new_df)

    add_to_excel(new_df, crypto_pair)

    if train_model_checkbox:
        model, mse, mae, r2 = train_model(new_df, variable1, variable2)
        st.success("Model trained successfully!", icon="✅")
        st.write(f"MSE: {mse:.4f}")
        st.write(f"MAE: {mae:.4f}")
        st.write(f"R2: {r2:.4f}")
