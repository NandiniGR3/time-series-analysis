# ============================================================
# PROGRAM 6
# Aim: To Estimate and Eliminate Trend in a Time Series
#      using Aggregation, Smoothing, and Polynomial Fitting
# Dataset: DailyDelhiClimateTest.csv
#============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------
# Step 1 : Load Dataset
# ------------------------------------------------------------

# VS Code / Jupyter
file_path = "DailyDelhiClimateTest.csv"

# If inside datasets folder
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

ts_data = data["meantemp"]

print("=" * 60)
print("FIRST 5 RECORDS")
print("=" * 60)

print(ts_data.head())

# ------------------------------------------------------------
# Step 4 : Monthly Aggregation
# ------------------------------------------------------------

monthly_data = ts_data.resample("ME").mean()

# ------------------------------------------------------------
# Step 5 : Moving Average Smoothing
# ------------------------------------------------------------

smoothed = monthly_data.rolling(
    window=3,
    center=True
).mean()

# ------------------------------------------------------------
# Step 6 : Polynomial Trend (Degree = 2)
# ------------------------------------------------------------

monthly_data = monthly_data.dropna()

X = np.arange(len(monthly_data)).reshape(-1, 1)
y = monthly_data.values

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

model = LinearRegression()

model.fit(X_poly, y)

trend = model.predict(X_poly)

# ------------------------------------------------------------
# Step 7 : Remove Trend
# ------------------------------------------------------------

detrended = y - trend

# ------------------------------------------------------------
# Step 8 : Plot Results
# ------------------------------------------------------------

plt.figure(figsize=(14,10))

# Original Series
plt.subplot(2,2,1)
plt.plot(monthly_data)
plt.title("Original Monthly Series")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.grid(True)

# Smoothed Series
plt.subplot(2,2,2)
plt.plot(smoothed)
plt.title("3-Month Moving Average")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.grid(True)

# Polynomial Trend
plt.subplot(2,2,3)
plt.plot(monthly_data, label="Original")
plt.plot(monthly_data.index, trend,
         color="red",
         linewidth=2,
         label="Polynomial Trend")

plt.title("Polynomial Trend")
plt.legend()
plt.grid(True)

# Detrended Series
plt.subplot(2,2,4)
plt.plot(monthly_data.index,
         detrended,
         color="green")

plt.title("Detrended Series")
plt.xlabel("Date")
plt.ylabel("Residual")
plt.grid(True)

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# Step 9 : Display Trend Values
# ------------------------------------------------------------

trend_df = pd.DataFrame({
    "Original": y,
    "Trend": trend,
    "Detrended": detrended
}, index=monthly_data.index)

print("\n")
print("=" * 60)
print("TREND ANALYSIS")
print("=" * 60)

print(trend_df.head())

# ------------------------------------------------------------
# Step 10 : Summary Statistics
# ------------------------------------------------------------

print("\nSummary Statistics")
print("=" * 60)

print(trend_df.describe())

print("\nProgram Executed Successfully.")
