# ============================================================
# PROGRAM 13
# Aim:
# Perform Factor Analysis and Cluster Analysis
# Dataset : DailyDelhiClimateTest.csv
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
# Step 2 : Select Features for Factor Analysis
# ------------------------------------------------------------

features = data[
    [
        "meantemp",
        "humidity",
        "wind_speed",
        "meanpressure"
    ]
]

# ------------------------------------------------------------
# Step 3 : Standardize Data
# ------------------------------------------------------------

scaler = StandardScaler()

scaled_data = scaler.fit_transform(features)

# ------------------------------------------------------------
# Step 4 : Factor Analysis
# ------------------------------------------------------------

fa = FactorAnalysis(n_components=2)

factors = fa.fit_transform(scaled_data)

print("=" * 60)
print("FACTOR LOADINGS")
print("=" * 60)

loading_df = pd.DataFrame(
    fa.components_,
    columns=features.columns,
    index=["Factor 1", "Factor 2"]
)

print(loading_df)

# ------------------------------------------------------------
# Step 5 : Select Features for Cluster Analysis
# ------------------------------------------------------------

cluster_features = data[
    [
        "meantemp",
        "humidity",
        "wind_speed"
    ]
]

scaled_cluster = scaler.fit_transform(cluster_features)

# ------------------------------------------------------------
# Step 6 : Apply K-Means Clustering
# ------------------------------------------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(scaled_cluster)

data["Cluster"] = clusters

# ------------------------------------------------------------
# Step 7 : Display Cluster Counts
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("CLUSTER DISTRIBUTION")
print("=" * 60)

print(data["Cluster"].value_counts().sort_index())

# ------------------------------------------------------------
# Step 8 : Scatter Plot
# ------------------------------------------------------------

plt.figure(figsize=(8,6))

for i in range(3):

    cluster_data = data[data["Cluster"] == i]

    plt.scatter(
        cluster_data["meantemp"],
        cluster_data["humidity"],
        label=f"Cluster {i}"
    )

plt.title("K-Means Cluster Analysis")

plt.xlabel("Mean Temperature")

plt.ylabel("Humidity")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# Step 9 : Display First Few Records
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("DATA WITH CLUSTER LABELS")
print("=" * 60)

print(data.head())

# ------------------------------------------------------------
# Step 10 : Completion Message
# ------------------------------------------------------------

print("\nProgram Executed Successfully.")
