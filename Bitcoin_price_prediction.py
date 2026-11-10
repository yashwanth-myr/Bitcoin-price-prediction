import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r'C:\Users\srika\OneDrive\Documents\Bitcoin_dataset_visualization\Bitcoin_investment\Bitcoin Historical Data (1).csv')  # Change to your actual file name

df['Date'] = pd.to_datetime(df['Date'])  # Convert Date column to datetime

df = df.sort_values(by='Date')  # Ensure data is in order

# Clean numerical columns (remove commas, convert percentages)
numeric_columns = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
for col in numeric_columns:
    df[col] = df[col].astype(str).str.replace(',', '')  # Remove commas
    if col == 'Change %':
        df[col] = df[col].str.replace('%', '').astype(float)  # Remove %
    elif col == 'Vol.':
        df[col] = df[col].str.replace('K', '').astype(float) * 1000  # Convert K to number
    else:
        df[col] = df[col].astype(float)  # Convert to float

# Create lag features (using past 60 days to predict next day)
lookback_days = 60
for i in range(1, lookback_days + 1):
    df[f'Price_Lag_{i}'] = df['Price'].shift(i)

df.dropna(inplace=True)  # Drop rows with NaN values after shifting

# Define features and target
features = [col for col in df.columns if 'Lag' in col]
target = 'Price'  # Predict next day's price

# Split into training and testing sets
train_size = int(0.8 * len(df))  # 80% train, 20% test
X_train, X_test = df[features][:train_size], df[features][train_size:]
y_train, y_test = df[target][:train_size], df[target][train_size:]

# Train XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# Predict next day's price
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae:.2f}')

# Check if investing today is profitable
last_features = df[features].iloc[-1].values.reshape(1, -1)
next_day_price = model.predict(last_features)[0]
current_price = df['Price'].iloc[-1]

print(f'Current Price: {current_price:.2f}, Predicted Next Day Price: {next_day_price:.2f}')
if next_day_price > current_price:
    print("Recommendation: BUY (Price expected to increase)")
else:
    print("Recommendation: DON'T BUY (Price expected to decrease or stay same)")

# Plot actual vs predicted prices
plt.figure(figsize=(10, 5))
plt.plot(df['Date'][train_size:], y_test, label='Actual Price', color='blue')
plt.plot(df['Date'][train_size:], y_pred, label='Predicted Price', color='red', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.title('Bitcoin Price Prediction')
plt.show()