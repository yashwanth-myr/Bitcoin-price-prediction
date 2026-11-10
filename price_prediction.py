import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from catboost import CatBoostRegressor
from prophet import Prophet

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file, thousands=',')
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.sort_values(by='Date')
    return df

def lstm_model(df):
    st.subheader("Bitcoin Price Prediction with LSTM")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['Price'] = scaler.fit_transform(df[['Price']])
    
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)
    
    seq_length = 60
    price_data = df['Price'].values.reshape(-1, 1)
    X, y = create_sequences(price_data, seq_length)
    
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    
    last_60_days = price_data[-seq_length:].reshape(1, seq_length, 1)
    predicted_prices = []
    future_dates = pd.date_range(start='2025-01-30', periods=3, freq='D')
    
    for i in range(3):
        next_price = model.predict(last_60_days)[0, 0]
        predicted_prices.append(next_price)
        last_60_days = np.roll(last_60_days, -1)
        last_60_days[0, -1, 0] = next_price
    
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
    predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predicted_prices.flatten()})
    st.dataframe(predictions_df)

def catboost_model(df):
    st.subheader("Bitcoin Price Prediction with CatBoost")
    
    df['MA5'] = df['Price'].rolling(window=5).mean()
    df['MA20'] = df['Price'].rolling(window=20).mean()
    df['Price_Momentum'] = df['Price'].diff()
    df['Volatility'] = df['Price'].rolling(window=5).std()
    df.dropna(inplace=True)
    
    features = ['MA5', 'MA20', 'Price_Momentum', 'Volatility']
    target = 'Price'
    
    X = df[features]
    y = df[target]
    
    train_size = int(0.8 * len(df))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    
    model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, loss_function='MAE', verbose=0)
    model.fit(X_train_scaled, y_train_scaled)
    
    last_features = df[features].iloc[-1].values.reshape(1, -1)
    future_dates = pd.date_range(start='2025-01-30', periods=3, freq='D')
    predicted_prices = model.predict(scaler_X.transform(last_features))
    predicted_prices = scaler_y.inverse_transform(predicted_prices.reshape(-1, 1))
    
    predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predicted_prices.flatten()})
    st.dataframe(predictions_df)

def prophet_model(df):
    st.subheader("Bitcoin Price Prediction with Prophet")
    
    df_prophet = df[['Date', 'Price']].rename(columns={'Date': 'ds', 'Price': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    
    future_dates = pd.date_range(start='2025-01-30', periods=3, freq='D')
    future_df = pd.DataFrame({'ds': future_dates})
    forecast = model.predict(future_df)
    
    predictions_df = forecast[['ds', 'yhat']]
    predictions_df.columns = ['Date', 'Predicted Price']
    st.dataframe(predictions_df)

st.sidebar.title("Select Prediction Model")
option = st.sidebar.radio("Choose an algorithm:", ["LSTM", "CatBoost", "Prophet"])

uploaded_file = st.file_uploader("Upload Bitcoin Data CSV", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)
    if option == "LSTM":
        lstm_model(df)
    elif option == "CatBoost":
        catboost_model(df)
    elif option == "Prophet":
        prophet_model(df)
