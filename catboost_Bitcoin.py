import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Streamlit UI
st.title("Bitcoin Price Prediction with CatBoost")

# File uploader
uploaded_file = st.file_uploader("Upload Bitcoin Data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, thousands=',')
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.sort_values(by='Date')
    
    # Feature engineering
    df['MA5'] = df['Price'].rolling(window=5).mean()
    df['MA20'] = df['Price'].rolling(window=20).mean()
    df['Price_Momentum'] = df['Price'].diff()
    df['Volatility'] = df['Price'].rolling(window=5).std()
    # Convert 'Vol.' column to numeric (handle 'K' suffix)
    df['Vol.'] = df['Vol.'].astype(str).apply(lambda x: float(x.replace('K', '')) * 1000 if 'K' in x else float(x.replace(',', '')))

    # Now it's safe to compute moving average
    df['Volume_MA5'] = df['Vol.'].rolling(window=5).mean()

    
    lookback_days = 30
    for i in range(1, lookback_days + 1):
        df[f'Price_Lag_{i}'] = df['Price'].shift(i)
        df[f'Volume_Lag_{i}'] = df['Vol.'].shift(i)
        if i <= 5:
            # Convert 'Change %' column to numeric
            df['Change %'] = df['Change %'].astype(str).str.replace('%', '').astype(float)

# Now it's safe to create lag features
            for i in range(1, lookback_days + 1):
                df[f'Price_Lag_{i}'] = df['Price'].shift(i)
                df[f'Volume_Lag_{i}'] = df['Vol.'].shift(i)
                if i <= 5:
                    df[f'Return_Lag_{i}'] = df['Change %'].shift(i)  # Now 'Change %' is numeric

    
    df.dropna(inplace=True)
    
    # Define features and target
    features = [col for col in df.columns if 'Lag' in col] + ['MA5', 'MA20', 'Price_Momentum', 'Volatility', 'Volume_MA5']
    target = 'Price'
    
    # Train-test split
    train_size = int(0.8 * len(df))
    X_train, X_test = df[features][:train_size], df[features][train_size:]
    y_train, y_test = df[target][:train_size], df[target][train_size:]
    
    # Scale features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    
    # Train CatBoost model
    model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, loss_function='MAE', verbose=0)
    model.fit(X_train_scaled, y_train_scaled)
    
    # Predict next 3 days
    last_features = df[features].iloc[-1].values.reshape(1, -1)
    future_dates = pd.to_datetime(['30-01-2025', '31-01-2025', '01-02-2025'], format='%d-%m-%Y')
    predicted_prices = []
    
    for i in range(3):
        last_features_scaled = scaler_X.transform(last_features)
        next_day_price_scaled = model.predict(last_features_scaled)
        next_day_price = scaler_y.inverse_transform(np.array(next_day_price_scaled).reshape(-1, 1))[0, 0]
        predicted_prices.append(next_day_price)
        
        last_features = np.roll(last_features, -1)
        last_features[0, -1] = next_day_price  # Update latest price
    
    # Create DataFrame for predictions
    predictions_df = pd.DataFrame({
        'Date': future_dates.strftime('%d-%m-%Y'),
        'Predicted Price': predicted_prices
    })
    
    st.write("### Predicted Bitcoin Prices for Next 3 Days")
    st.dataframe(predictions_df)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'][-100:], df['Price'][-100:], label='Actual Price')
    plt.plot(future_dates, predicted_prices, 'ro', label='Predicted Price')
    plt.xlabel('Date')
    plt.ylabel('Bitcoin Price')
    plt.legend()
    plt.title('Bitcoin Price Prediction with CatBoost')
    st.pyplot(plt)
