# ============================================================
# PROGRAM 2
# Aim: To Load and Handle Time Series Data
# Dataset: Fashion_Retail_Sales.csv
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Step 1: Load Dataset
# ------------------------------------------------------------

# For VS Code / Jupyter
file_path = "Fashion_Retail_Sales.csv"

# If dataset is inside a folder:
# file_path = "datasets/Fashion_Retail_Sales.csv"

# For Google Colab:
# file_path = "/content/Fashion_Retail_Sales.csv"

try:
    data = pd.read_csv(file_path)
    print("Dataset Loaded Successfully.\n")
except FileNotFoundError:
    print("Error: Dataset not found.")
    exit()

# ------------------------------------------------------------
# Step 2: Display Original Dataset
# ------------------------------------------------------------

print("=" * 60)
print("FIRST 5 RECORDS")
print("=" * 60)

print(data.head())

print("\nDataset Shape :", data.shape)

# ------------------------------------------------------------
# Step 3: Remove Extra Spaces from Column Names
# ------------------------------------------------------------

data.columns = data.columns.str.strip()

# ------------------------------------------------------------
# Step 4: Convert Date Column
# ------------------------------------------------------------

if "Date Purchase" in data.columns:

    data["Date Purchase"] = pd.to_datetime(
        data["Date Purchase"],
        format="%d-%m-%Y",
        errors="coerce"
    )

    # Fill missing dates using Forward Fill
    data["Date Purchase"] = data["Date Purchase"].ffill()

else:
    print("Date Purchase column not found.")
    exit()

# ------------------------------------------------------------
# Step 5: Set Date as Time Series Index
# ------------------------------------------------------------

data.set_index("Date Purchase", inplace=True)

# ------------------------------------------------------------
# Step 6: Sort Data Chronologically
# ------------------------------------------------------------

data.sort_index(inplace=True)

print("\nTime Series Index Created Successfully.")

# ------------------------------------------------------------
# Step 7: Display Dataset Information
# ------------------------------------------------------------

print("\nDataset Information")
print("=" * 60)

print(data.info())

# ------------------------------------------------------------
# Step 8: Monthly Sales Aggregation
# ------------------------------------------------------------

monthly_sales = data["Purchase Amount (USD)"].resample("ME").sum()

print("\nMonthly Sales")
print("=" * 60)

print(monthly_sales.head(10))

# ------------------------------------------------------------
# Step 9: Plot Time Series
# ------------------------------------------------------------

plt.figure(figsize=(12,6))

plt.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o",
    linewidth=2
)

plt.title("Monthly Sales Time Series")
plt.xlabel("Month")
plt.ylabel("Sales (USD)")
plt.grid(True)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Step 10: Summary Statistics
# ------------------------------------------------------------

print("\nSummary Statistics")
print("=" * 60)

print(monthly_sales.describe())

print("\nProgram Executed Successfully.")
