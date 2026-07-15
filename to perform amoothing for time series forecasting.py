# ============================================================
# PROGRAM 9
# Aim: To Perform Smoothing for Time Series Forecasting
#
# Dataset : DailyDelhiClimateTest.csv
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    ExponentialSmoothing
)

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

# ------------------------------------------------------------
# Step 3 : Select Time Series
# ------------------------------------------------------------

ts_data = data["meantemp"].dropna()

print("=" * 60)
print("FIRST 5 RECORDS")
print("=" * 60)

print(ts_data.head())

# ------------------------------------------------------------
# Step 4 : Simple Moving Average (SMA)
# ------------------------------------------------------------

window_size = 30

sma = ts_data.rolling(window=window_size).mean()

# ------------------------------------------------------------
# Step 5 : Simple Exponential Smoothing (SES)
# ------------------------------------------------------------

ses_model = SimpleExpSmoothing(ts_data).fit(
    smoothing_level=0.2,
    optimized=False
)

ses = ses_model.fittedvalues

# ------------------------------------------------------------
# Step 6 : Holt's Linear Trend Method
# ------------------------------------------------------------

holt_model = ExponentialSmoothing(
    ts_data,
    trend="add",
    seasonal=None
).fit()

holt = holt_model.fittedvalues

# ------------------------------------------------------------
# Step 7 : Display Sample Results
# ------------------------------------------------------------

result = pd.DataFrame({

    "Original": ts_data,

    "Moving Average": sma,

    "SES": ses,

    "Holt": holt

})

print("\n")
print("=" * 60)
print("SMOOTHING RESULTS")
print("=" * 60)

print(result.tail(10))

# ------------------------------------------------------------
# Step 8 : Plot Comparison
# ------------------------------------------------------------

plt.figure(figsize=(14,6))

plt.plot(
    ts_data,
    label="Original Data",
    linewidth=2
)

plt.plot(
    sma,
    label="30-Day Moving Average"
)

plt.plot(
    ses,
    label="Simple Exponential Smoothing"
)

plt.plot(
    holt,
    label="Holt Linear Trend"
)

plt.title("Smoothing Techniques for Time Series Forecasting")

plt.xlabel("Date")
plt.ylabel("Mean Temperature")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# Step 9 : Summary Statistics
# ------------------------------------------------------------

print("\nSummary Statistics")
print("=" * 60)

print(result.describe())

# ------------------------------------------------------------
# Step 10 : Completion Message
# ------------------------------------------------------------

print("\nProgram Executed Successfully.")
