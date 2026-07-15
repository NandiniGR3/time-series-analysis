# ============================================================
# PROGRAM 8
# Aim : To Perform Time Series Forecasting using
#       1. Moving Average
#       2. Exponential Smoothing
#       3. ARIMA
#
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

# VS Code / Jupyter
file_path = "DailyDelhiClimateTest.csv"

# If dataset is inside datasets folder
# file_path = "datasets/DailyDelhiClimateTest.csv"

# Google Colab
# file_path = "/content/DailyDelhiClimateTest.csv"

try:
    data = pd.read_csv(file_path)
    print("Dataset Loaded Successfully.\n")
except FileNotFoundError:
    print("Dataset not found.")
    exit()

# ------------------------------------------------------------
# Step 2 : Convert Date Column
# ------------------------------------------------------------

data["date"] = pd.to_datetime(data["date"])

data.set_index("date", inplace=True)

# ------------------------------------------------------------
# Step 3 : Select Time Series
# ------------------------------------------------------------

ts_data = data["meantemp"].dropna()

print("="*60)
print("FIRST 5 RECORDS")
print("="*60)

print(ts_data.head())

# ------------------------------------------------------------
# Step 4 : Train-Test Split
# ------------------------------------------------------------

train_size = int(len(ts_data) * 0.80)

train = ts_data[:train_size]
test = ts_data[train_size:]

print("\nTraining Samples :", len(train))
print("Testing Samples  :", len(test))

# ------------------------------------------------------------
# Step 5 : Moving Average Forecast
# ------------------------------------------------------------

window = 7

moving_average = train.rolling(window=window).mean()

last_average = moving_average.iloc[-1]

ma_forecast = pd.Series(
    [last_average] * len(test),
    index=test.index
)

# ------------------------------------------------------------
# Step 6 : Exponential Smoothing
# ------------------------------------------------------------

exp_model = ExponentialSmoothing(
    train,
    trend="add",
    seasonal=None
)

exp_fit = exp_model.fit()

exp_forecast = exp_fit.forecast(len(test))

# ------------------------------------------------------------
# Step 7 : ARIMA Model
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
# Step 8 : Model Evaluation
# ------------------------------------------------------------

def evaluate(actual, predicted):

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)

    return mae, rmse

ma_mae, ma_rmse = evaluate(test, ma_forecast)
exp_mae, exp_rmse = evaluate(test, exp_forecast)
arima_mae, arima_rmse = evaluate(test, arima_forecast)

print("\n")
print("="*60)
print("MODEL PERFORMANCE")
print("="*60)

print(f"Moving Average        MAE = {ma_mae:.3f}   RMSE = {ma_rmse:.3f}")
print(f"Exponential Smoothing MAE = {exp_mae:.3f}   RMSE = {exp_rmse:.3f}")
print(f"ARIMA                 MAE = {arima_mae:.3f}   RMSE = {arima_rmse:.3f}")

# ------------------------------------------------------------
# Step 9 : Plot Forecast Comparison
# ------------------------------------------------------------

plt.figure(figsize=(14,6))

plt.plot(train,
         label="Training Data")

plt.plot(test,
         label="Actual Data",
         linewidth=2)

plt.plot(test.index,
         ma_forecast,
         label="Moving Average Forecast")

plt.plot(test.index,
         exp_forecast,
         label="Exponential Smoothing")

plt.plot(test.index,
         arima_forecast,
         label="ARIMA Forecast",
         linewidth=2)

plt.title("Time Series Forecast Comparison")

plt.xlabel("Date")
plt.ylabel("Mean Temperature")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# Step 10 : Forecast Table
# ------------------------------------------------------------

forecast_df = pd.DataFrame({

    "Actual": test,

    "Moving Average": ma_forecast,

    "Exponential Smoothing": exp_forecast,

    "ARIMA": arima_forecast

})

print("\n")
print("="*60)
print("FORECAST RESULTS")
print("="*60)

print(forecast_df.head(10))

print("\nProgram Executed Successfully.")
