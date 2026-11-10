# utils.py
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def preprocess_data(df):
    # Clean Volume column (handle 'K' suffix)
    df['Vol.'] = df['Vol.'].astype(str).apply(lambda x: float(x.replace('K', '')) * 1000 if 'K' in x else float(x))

    # Clean Change % column
    df['Change %'] = df['Change %'].astype(str).str.replace('%', '').astype(float)

    # Additional technical indicators
    df['MA5'] = df['Price'].rolling(window=5).mean()
    df['MA20'] = df['Price'].rolling(window=20).mean()
    df['Price_Momentum'] = df['Price'].diff()
    df['Volatility'] = df['Price'].rolling(window=5).std()
    df['Volume_MA5'] = df['Vol.'].rolling(window=5).mean()

    # Create lag features
    lookback_days = 30
    for i in range(1, lookback_days + 1):
        df[f'Price_Lag_{i}'] = df['Price'].shift(i)
        df[f'Volume_Lag_{i}'] = df['Vol.'].shift(i)
        if i <= 5:
            df[f'Return_Lag_{i}'] = df['Change %'].shift(i)

    df.dropna(inplace=True)
    return df