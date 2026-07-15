# ============================================================
# PROGRAM 7
# Aim: To Eliminate Trend and Seasonality in a Time Series
#      using Differencing and Decomposition Techniques
#
# Dataset : DailyDelhiClimateTest.csv
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose

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
    print("Dataset not found.")
    exit()

# ------------------------------------------------------------
# Step 2 : Convert Date Column
# ------------------------------------------------------------

data["date"] = pd.to_datetime(data["date"])

# Set Date as Index
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
# Step 4 : Plot Original Series
# ------------------------------------------------------------

plt.figure(figsize=(12,5))

plt.plot(ts_data)

plt.title("Original Time Series")
plt.xlabel("Date")
plt.ylabel("Mean Temperature")

plt.grid(True)

plt.show()

# ------------------------------------------------------------
# Step 5 : First Order Differencing
# ------------------------------------------------------------

first_difference = ts_data.diff().dropna()

# ------------------------------------------------------------
# Step 6 : Seasonal Differencing
# ------------------------------------------------------------

seasonal_period = 7      # Weekly Seasonality

seasonal_difference = first_difference.diff(
    seasonal_period
).dropna()

# ------------------------------------------------------------
# Step 7 : Seasonal Decomposition
# ------------------------------------------------------------

decomposition = seasonal_decompose(
    ts_data,
    model="additive",
    period=7
)

# ------------------------------------------------------------
# Step 8 : Plot Decomposition
# ------------------------------------------------------------

decomposition.plot()

plt.suptitle(
    "Seasonal Decomposition",
    fontsize=16
)

plt.show()

# ------------------------------------------------------------
# Step 9 : Plot Original vs Processed Series
# ------------------------------------------------------------

plt.figure(figsize=(14,6))

# Original Series
plt.subplot(1,2,1)

plt.plot(ts_data)

plt.title("Original Time Series")
plt.grid(True)

# Processed Series
plt.subplot(1,2,2)

plt.plot(
    seasonal_difference,
    color="green"
)

plt.title("After Removing Trend & Seasonality")

plt.grid(True)

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# Step 10 : Display Sample Data
# ------------------------------------------------------------

result = pd.DataFrame({

    "Original": ts_data,

    "First Difference": first_difference,

    "Seasonal Difference": seasonal_difference

})

print("\n")
print("=" * 60)
print("TRANSFORMED DATA")
print("=" * 60)

print(result.tail(10))

print("\nSummary Statistics")
print("=" * 60)

print(result.describe())

print("\nProgram Executed Successfully.")
