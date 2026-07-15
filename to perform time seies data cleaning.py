# ============================================================
# PROGRAM 1
# Aim: To Perform Time Series Data Cleaning
# Dataset: Fashion_Retail_Sales.csv
# ============================================================

import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Step 1: Load Dataset
# ------------------------------------------------------------

# For VS Code / Jupyter (dataset in same folder)
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
    print("Place 'Fashion_Retail_Sales.csv' in the project folder.")
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
# Step 4: Check Missing Values
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES BEFORE CLEANING")
print("=" * 60)

print(data.isnull().sum())

# ------------------------------------------------------------
# Step 5: Fill Missing Values
# ------------------------------------------------------------

for column in data.columns:

    if data[column].dtype == "object":

        # Fill categorical values using Mode
        mode_value = data[column].mode()

        if len(mode_value) > 0:
            data[column] = data[column].fillna(mode_value[0])

    else:

        # Fill numerical values using Median
        median_value = data[column].median()

        data[column] = data[column].fillna(median_value)

# ------------------------------------------------------------
# Step 6: Convert Date Column
# ------------------------------------------------------------

if "Date Purchase" in data.columns:

    data["Date Purchase"] = pd.to_datetime(
        data["Date Purchase"],
        format="%d-%m-%Y",
        errors="coerce"
    )

    # Forward fill missing dates
    data["Date Purchase"] = data["Date Purchase"].ffill()

else:
    print("\nWarning: 'Date Purchase' column not found.")

# ------------------------------------------------------------
# Step 7: Remove Duplicate Records
# ------------------------------------------------------------

duplicates_before = data.duplicated().sum()

data = data.drop_duplicates()

duplicates_after = data.duplicated().sum()

# ------------------------------------------------------------
# Step 8: Final Dataset Information
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET INFORMATION AFTER CLEANING")
print("=" * 60)

print(data.info())

print("\n" + "=" * 60)
print("MISSING VALUES AFTER CLEANING")
print("=" * 60)

print(data.isnull().sum())

print("\n" + "=" * 60)
print("DUPLICATE RECORDS")
print("=" * 60)

print("Duplicates Before Removing :", duplicates_before)
print("Duplicates After Removing  :", duplicates_after)

print("\n" + "=" * 60)
print("CLEANED DATASET (FIRST 5 ROWS)")
print("=" * 60)

print(data.head())

print("\nDataset Shape After Cleaning :", data.shape)

# ------------------------------------------------------------
# Step 9: Save Cleaned Dataset
# ------------------------------------------------------------

output_file = "Fashion_Retail_Sales_Cleaned.csv"

data.to_csv(output_file, index=False)

print(f"\nCleaned dataset saved as '{output_file}'")

print("\nProgram Executed Successfully.")
