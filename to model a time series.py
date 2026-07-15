# ============================================================
# PROGRAM 11
# Aim: To Model a Time Series using
#      1. Moving Average Model
#      2. Exponential Smoothing
#      3. ARIMA Model
# Dataset : DailyDelhiClimateTest.csv
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

# ------------------------------------------------------------
# Step 1 : Load Dataset
# ------------------------------------------------------------

# For VS Code / Jupyter
file_path = "DailyDelhiClimateTest.csv"

# If dataset is inside datasets folder
# file_path = "datasets/DailyDelhiClimateTest.csv"

# For Google Colab
# file_path = "/content/DailyDelhiClimateTest.csv"

try:
    data = pd.read_csv(file_path)
    print("Dataset Loaded Successfully.\n")
except FileNotFoundError:
    print("Error: Dataset not found.")
    exit()

# ------------------------------------------------------------
# Step 2 : Convert Date Column
# ------------------------------------------------------------

data["date"] = pd.to_datetime(data["date"])
data.set_index("date", inplace=True)

# Select Mean Temperature
ts_data = data["meantemp"].ffill()

# ------------------------------------------------------------
# Step 3 : Train-Test Split
# ------------------------------------------------------------

train_size = int(len(ts_data) * 0.80)

train = ts_data[:train_size]
test = ts_data[train_size:]

print("=" * 60)
print("TRAIN TEST INFORMATION")
print("=" * 60)

print("Training Samples :", len(train))
print("Testing Samples  :", len(test))

# ------------------------------------------------------------
# Step 4 : Moving Average Model
# ------------------------------------------------------------

window = 7

ma_model = train.rolling(window=window).mean()

# Forecast using last moving average value
last_ma = ma_model.dropna().iloc[-1]

ma_forecast = np.repeat(last_ma, len(test))

# ------------------------------------------------------------
# Step 5 : Exponential Smoothing Model
# ------------------------------------------------------------

exp_model = ExponentialSmoothing(
    train,
    trend="add",
    seasonal=None
).fit()

exp_forecast = exp_model.forecast(len(test))

# ------------------------------------------------------------
# Step 6 : ARIMA Model
# ------------------------------------------------------------

arima_model = ARIMA(
    train,
    order=(1,1,1)
)

arima_fit = arima_model.fit()

arima_forecast = arima_fit.forecast(
    steps=len(test)
)

# ------------------------------------------------------------
# Step 7 : Model Evaluation
# ------------------------------------------------------------

def evaluate(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse

ma_mae, ma_rmse = evaluate(test, ma_forecast)
exp_mae, exp_rmse = evaluate(test, exp_forecast)
arima_mae, arima_rmse = evaluate(test, arima_forecast)

print("\n")
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Moving Average        MAE = {ma_mae:.3f}   RMSE = {ma_rmse:.3f}")
print(f"Exponential Smoothing MAE = {exp_mae:.3f}   RMSE = {exp_rmse:.3f}")
print(f"ARIMA                 MAE = {arima_mae:.3f}   RMSE = {arima_rmse:.3f}")

# ------------------------------------------------------------
# Step 8 : Plot Comparison
# ------------------------------------------------------------

plt.figure(figsize=(12,6))

plt.plot(train, label="Training Data")

plt.plot(test, label="Actual Data", linewidth=2)

plt.plot(
    test.index,
    ma_forecast,
    label="Moving Average Forecast"
)

plt.plot(
    test.index,
    exp_forecast,
    label="Exponential Smoothing Forecast"
)

plt.plot(
    test.index,
    arima_forecast,
    label="ARIMA Forecast"
)

plt.title("Time Series Modelling Comparison")

plt.xlabel("Date")
plt.ylabel("Temperature")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# Step 9 : Forecast Table
# ------------------------------------------------------------

forecast_df = pd.DataFrame({

    "Actual": test,

    "Moving Average": ma_forecast,

    "Exponential Smoothing": exp_forecast,

    "ARIMA": arima_forecast

})

print("\n")
print("=" * 60)
print("FORECAST RESULTS")
print("=" * 60)

print(forecast_df.head(10))

# ------------------------------------------------------------
# Step 10 : Completion Message
# ------------------------------------------------------------

print("\nProgram Executed Successfully.")
