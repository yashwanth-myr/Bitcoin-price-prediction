import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Streamlit UI
st.title("Bitcoin Price Prediction with LSTM")

# File uploader
uploaded_file = st.file_uploader("Upload Bitcoin Data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, thousands=',')
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.sort_values(by='Date')
    df.set_index('Date', inplace=True)
    
    # Normalize prices
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['Price'] = scaler.fit_transform(df[['Price']])
    
    # Prepare data for LSTM
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)
    
    seq_length = 60  # Use past 60 days to predict next day
    price_data = df['Price'].values.reshape(-1, 1)
    X, y = create_sequences(price_data, seq_length)
    
    # Train-test split
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Define LSTM model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    
    # Predict next 3 days
    last_60_days = price_data[-seq_length:].reshape(1, seq_length, 1)
    predicted_prices = []
    future_dates = pd.to_datetime(['30-01-2025', '31-01-2025', '01-02-2025'], format='%d-%m-%Y')

    
    for i in range(3):
        next_price = model.predict(last_60_days)[0, 0]
        predicted_prices.append(next_price)
        last_60_days = np.roll(last_60_days, -1)
        last_60_days[0, -1, 0] = next_price
    
    # Inverse transform predictions
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
    
    # Create DataFrame for predictions
    predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': predicted_prices.flatten()
    })
    
    st.write("### Predicted Bitcoin Prices for Next 3 Days")
    st.dataframe(predictions_df)
    
    # Plot predictions
    plt.figure(figsize=(10, 5))
    plt.plot(df.index[-100:], scaler.inverse_transform(df['Price'].values[-100:].reshape(-1, 1)), label='Actual Price')
    plt.plot(future_dates, predicted_prices, 'ro', label='Predicted Price')
    plt.xlabel('Date')
    plt.ylabel('Bitcoin Price')
    plt.legend()
    plt.title('Bitcoin Price Prediction with LSTM')
    st.pyplot(plt)
