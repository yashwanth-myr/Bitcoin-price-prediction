import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from utils import preprocess_data


@st.cache_resource
def train_model(X_train_scaled, y_train_scaled):
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train_scaled)
    return model


def run_model(df):
    # Streamlit UI
    st.title('Bitcoin Price Prediction App (Random Forest)')
    df = preprocess_data(df)
    # Define features
    price_lags = [col for col in df.columns if 'Price_Lag' in col]
    volume_lags = [col for col in df.columns if 'Volume_Lag' in col]
    return_lags = [col for col in df.columns if 'Return_Lag' in col]
    technical_indicators = ['MA5', 'MA20', 'Price_Momentum', 'Volatility', 'Volume_MA5']

    features = price_lags + volume_lags + return_lags + technical_indicators
    target = 'Price'

    # Split into training and testing sets
    train_size = int(0.8 * len(df))
    X_train, X_test = df[features][:train_size], df[features][train_size:]
    y_train, y_test = df[target][:train_size], df[target][train_size:]

    # Scale the features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler_X.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler_X.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    # Train Random Forest model
    model = train_model(X_train_scaled, y_train_scaled)

    # Predict
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # Evaluate model
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    st.write(f'Mean Absolute Error: ${mae:,.2f}')
    st.write(f'Mean Absolute Percentage Error: {mape:.2f}%')

    # Predict next three days' prices
    last_features = df[features].iloc[-1].values.reshape(1, -1)
    last_date = df['Date'].iloc[-1]

    predictions = []
    for i in range(1, 4):  # Predict for the next 3 days
        last_features_scaled = pd.DataFrame(
            scaler_X.transform(last_features),
            columns=df[features].columns
        )
        next_day_price_scaled = model.predict(last_features_scaled)
        next_day_price = scaler_y.inverse_transform(next_day_price_scaled.reshape(-1, 1))[0][0]

        # Update the date
        next_date = last_date + pd.Timedelta(days=i)

        # Store the prediction
        predictions.append({
            'Date': next_date,
            'Predicted Price': next_day_price
        })

        # Update last_features for the next prediction
        last_features = np.roll(last_features, -1)
        last_features[0, -1] = next_day_price  # Update the latest price in the features

    # Display predictions for the next three days
    st.write("Predictions for the next three days:")
    for pred in predictions:
        st.write(f"Date: {pred['Date'].strftime('%d-%m-%Y')}, Predicted Price: ${pred['Predicted Price']:,.2f}")

    # Plot actual vs predicted prices with confidence bands
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot historical actual and predicted prices
    ax.plot(df['Date'][train_size:], y_test, label='Actual Price', color='blue')
    ax.plot(df['Date'][train_size:], y_pred, label='Predicted Price', color='red', linestyle='--')

    # Add confidence bands (using standard deviation of predictions)
    std_dev = np.std(y_pred - y_test)
    ax.fill_between(df['Date'][train_size:],
                   y_pred - 2 * std_dev,
                   y_pred + 2 * std_dev,
                   color='red',
                   alpha=0.1,
                   label='95% Confidence Interval')

    # Plot the next three days' predictions
    pred_dates = [pred['Date'] for pred in predictions]
    pred_prices = [pred['Predicted Price'] for pred in predictions]
    ax.plot(pred_dates, pred_prices, 'go-', label='Next 3 Days Predictions', markersize=8)  # Green markers for predictions

    # Annotate the predicted points
    for i, (date, price) in enumerate(zip(pred_dates, pred_prices)):
        ax.annotate(f"Day {i + 1}: ${price:,.2f}",
                    (date, price),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center')

    ax.set_xlabel('Date')
    ax.set_ylabel('Bitcoin Price ($)')
    ax.legend()
    ax.set_title('Bitcoin Price Prediction with Random Forest (Including Next 3 Days)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)