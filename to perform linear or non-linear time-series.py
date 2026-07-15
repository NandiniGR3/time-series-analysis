# ============================================================
# PROGRAM 10
# Aim: To Check Linear and Non-Linear Trends in a Time Series
# Dataset : DailyDelhiClimateTest.csv
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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
# Step 4 : Create Time Index
# ------------------------------------------------------------

X = np.arange(len(ts_data)).reshape(-1, 1)
y = ts_data.values

# ------------------------------------------------------------
# Step 5 : Linear Trend Model
# ------------------------------------------------------------

linear_model = LinearRegression()

linear_model.fit(X, y)

linear_trend = linear_model.predict(X)

# ------------------------------------------------------------
# Step 6 : Polynomial (Non-Linear) Trend Model
# ------------------------------------------------------------

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

poly_model = LinearRegression()

poly_model.fit(X_poly, y)

poly_trend = poly_model.predict(X_poly)

# ------------------------------------------------------------
# Step 7 : Display Trend Values
# ------------------------------------------------------------

trend_df = pd.DataFrame({
    "Original": y,
    "Linear Trend": linear_trend,
    "Polynomial Trend": poly_trend
}, index=ts_data.index)

print("\n")
print("=" * 60)
print("TREND ANALYSIS")
print("=" * 60)

print(trend_df.head(10))

# ------------------------------------------------------------
# Step 8 : Plot Comparison
# ------------------------------------------------------------

plt.figure(figsize=(12,6))

plt.plot(
    ts_data,
    label="Original Data",
    linewidth=2
)

plt.plot(
    ts_data.index,
    linear_trend,
    color="red",
    linewidth=2,
    label="Linear Trend"
)

plt.plot(
    ts_data.index,
    poly_trend,
    color="green",
    linewidth=2,
    label="Polynomial Trend"
)

plt.title("Linear vs Non-Linear Trend")
plt.xlabel("Date")
plt.ylabel("Mean Temperature")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# Step 9 : Model Scores
# ------------------------------------------------------------

linear_score = linear_model.score(X, y)
poly_score = poly_model.score(X_poly, y)

print("\n")
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Linear Regression R² Score     : {linear_score:.4f}")
print(f"Polynomial Regression R² Score : {poly_score:.4f}")

# ------------------------------------------------------------
# Step 10 : Summary Statistics
# ------------------------------------------------------------

print("\nSummary Statistics")
print("=" * 60)

print(trend_df.describe())

print("\nProgram Executed Successfully.")
