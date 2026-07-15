# ============================================================
# PROGRAM 4
# Aim: To Check the Stationarity of a Time Series
# Dataset: DailyDelhiClimateTest.csv
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ------------------------------------------------------------
# Step 1: Load Dataset
# ------------------------------------------------------------

# For VS Code / Jupyter
file_path = "DailyDelhiClimateTest.csv"

# If dataset is inside a folder:
# file_path = "datasets/DailyDelhiClimateTest.csv"

# For Google Colab:
# file_path = "/content/DailyDelhiClimateTest.csv"

try:
    data = pd.read_csv(file_path)
    print("Dataset Loaded Successfully.\n")
except FileNotFoundError:
    print("Dataset not found.")
    exit()

# ------------------------------------------------------------
# Step 2: Convert Date Column
# ------------------------------------------------------------

data["date"] = pd.to_datetime(data["date"])

# Set Date as Index
data.set_index("date", inplace=True)

# ------------------------------------------------------------
# Step 3: Select Time Series Variable
# ------------------------------------------------------------

ts_data = data["meantemp"].dropna()

print("=" * 60)
print("FIRST 5 RECORDS")
print("=" * 60)

print(ts_data.head())

# ------------------------------------------------------------
# Step 4: Plot Original Time Series
# ------------------------------------------------------------

plt.figure(figsize=(12,5))

plt.plot(ts_data)

plt.title("Original Time Series")
plt.xlabel("Date")
plt.ylabel("Mean Temperature")

plt.grid(True)

plt.show()

# ------------------------------------------------------------
# Step 5: Augmented Dickey-Fuller Test
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("ADF TEST (ORIGINAL DATA)")
print("=" * 60)

result = adfuller(ts_data)

print(f"ADF Statistic : {result[0]:.6f}")
print(f"P-value       : {result[1]:.6f}")
print(f"Lags Used     : {result[2]}")
print(f"Observations  : {result[3]}")

print("\nCritical Values")

for key, value in result[4].items():
    print(f"{key} : {value:.6f}")

if result[1] < 0.05:
    print("\nResult : Time Series is Stationary")
else:
    print("\nResult : Time Series is NOT Stationary")

# ------------------------------------------------------------
# Step 6: First Order Differencing
# ------------------------------------------------------------

ts_diff = ts_data.diff().dropna()

# ------------------------------------------------------------
# Step 7: ADF Test After Differencing
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("ADF TEST AFTER DIFFERENCING")
print("=" * 60)

result2 = adfuller(ts_diff)

print(f"ADF Statistic : {result2[0]:.6f}")
print(f"P-value       : {result2[1]:.6f}")

if result2[1] < 0.05:
    print("\nResult : Stationarity Achieved")
else:
    print("\nResult : Still Non-Stationary")

# ------------------------------------------------------------
# Step 8: Plot Differenced Series
# ------------------------------------------------------------

plt.figure(figsize=(12,5))

plt.plot(ts_diff)

plt.title("Differenced Time Series")
plt.xlabel("Date")
plt.ylabel("Differenced Temperature")

plt.grid(True)

plt.show()

# ------------------------------------------------------------
# Step 9: ACF Plot
# ------------------------------------------------------------

plt.figure(figsize=(10,4))

plot_acf(ts_diff, lags=30)

plt.show()

# ------------------------------------------------------------
# Step 10: PACF Plot
# ------------------------------------------------------------

plt.figure(figsize=(10,4))

plot_pacf(
    ts_diff,
    lags=30,
    method="ywm"
)

plt.show()

print("\nProgram Executed Successfully.")
