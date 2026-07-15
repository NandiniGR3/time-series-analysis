# ============================================================
# PROGRAM 5
# Aim: To Make a Time Series Stationary
# Dataset: Fashion_Retail_Sales.csv
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# ------------------------------------------------------------
# Step 1: Load Dataset
# ------------------------------------------------------------

# For VS Code / Jupyter
file_path = "Fashion_Retail_Sales.csv"

# If dataset is inside datasets folder
# file_path = "datasets/Fashion_Retail_Sales.csv"

# For Google Colab
# file_path = "/content/Fashion_Retail_Sales.csv"

try:
    data = pd.read_csv(file_path)
    print("Dataset Loaded Successfully.\n")
except FileNotFoundError:
    print("Dataset not found.")
    exit()

# ------------------------------------------------------------
# Step 2: Basic Cleaning
# ------------------------------------------------------------

data.columns = data.columns.str.strip()

data["Date Purchase"] = pd.to_datetime(
    data["Date Purchase"],
    format="%d-%m-%Y",
    errors="coerce"
)

# Forward fill missing dates
data["Date Purchase"] = data["Date Purchase"].ffill()

# Remove duplicate records
data = data.drop_duplicates()

# ------------------------------------------------------------
# Step 3: Set Date as Index
# ------------------------------------------------------------

data.set_index("Date Purchase", inplace=True)

# Sort by Date
data.sort_index(inplace=True)

# ------------------------------------------------------------
# Step 4: Create Daily Time Series
# ------------------------------------------------------------

ts_data = data["Purchase Amount (USD)"].resample("D").sum()

# Remove zero values before log transformation
ts_data = ts_data.replace(0, np.nan).dropna()

print("=" * 60)
print("ORIGINAL TIME SERIES")
print("=" * 60)

print(ts_data.head())

# ------------------------------------------------------------
# Step 5: Plot Original Series
# ------------------------------------------------------------

plt.figure(figsize=(12,5))

plt.plot(ts_data)

plt.title("Original Time Series")
plt.xlabel("Date")
plt.ylabel("Purchase Amount (USD)")
plt.grid(True)

plt.show()

# ------------------------------------------------------------
# Step 6: Log Transformation
# ------------------------------------------------------------

ts_log = np.log(ts_data)

# ------------------------------------------------------------
# Step 7: First Differencing
# ------------------------------------------------------------

ts_diff1 = ts_log.diff().dropna()

# ------------------------------------------------------------
# Step 8: Second Differencing
# ------------------------------------------------------------

ts_diff2 = ts_diff1.diff().dropna()

# ------------------------------------------------------------
# Step 9: ADF Test
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("ADF TEST AFTER TRANSFORMATION")
print("=" * 60)

result = adfuller(ts_diff2)

print(f"ADF Statistic : {result[0]:.6f}")
print(f"P-value       : {result[1]:.6f}")
print(f"Lags Used     : {result[2]}")
print(f"Observations  : {result[3]}")

print("\nCritical Values")

for key, value in result[4].items():
    print(f"{key} : {value:.6f}")

if result[1] < 0.05:
    print("\nResult : Stationarity Achieved")
else:
    print("\nResult : Series is Still Non-Stationary")

# ------------------------------------------------------------
# Step 10: Plot Comparison
# ------------------------------------------------------------

plt.figure(figsize=(14,6))

plt.subplot(1,2,1)

plt.plot(ts_data)

plt.title("Original Series")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.grid(True)

plt.subplot(1,2,2)

plt.plot(ts_diff2)

plt.title("Stationary Series")
plt.xlabel("Date")
plt.ylabel("Differenced Log Sales")

plt.grid(True)

plt.tight_layout()

plt.show()

print("\nProgram Executed Successfully.")
