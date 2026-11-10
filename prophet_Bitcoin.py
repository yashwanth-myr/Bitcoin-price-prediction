import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Streamlit UI
st.title("Bitcoin Price Prediction with Prophet")

# File uploader
uploaded_file = st.file_uploader("Upload Bitcoin Data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, thousands=',')
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.sort_values(by='Date')
    
    # Prepare data for Prophet
    df_prophet = df[['Date', 'Price']].rename(columns={'Date': 'ds', 'Price': 'y'})
    
    # Train Prophet model
    model = Prophet()
    model.fit(df_prophet)
    
    # Create future dataframe for prediction
    future_dates = pd.date_range(start='2025-01-30', periods=3, freq='D')
    future_df = pd.DataFrame({'ds': future_dates})
    
    # Predict future prices
    forecast = model.predict(future_df)
    
    # Display predictions
    predictions_df = forecast[['ds', 'yhat']]
    predictions_df.columns = ['Date', 'Predicted Price']
    predictions_df['Date'] = predictions_df['Date'].dt.strftime('%d-%m-%Y')
    
    st.write("### Predicted Bitcoin Prices for Next 3 Days")
    st.dataframe(predictions_df)
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 5))
    model.plot(forecast, ax=ax)
    plt.xlabel('Date')
    plt.ylabel('Bitcoin Price')
    plt.title('Bitcoin Price Prediction with Prophet')
    st.pyplot(fig)
