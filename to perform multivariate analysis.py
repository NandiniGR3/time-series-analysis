# ============================================================
# PROGRAM 12
# Aim:
# Perform SARIMA Forecasting, Covariance,
# Correlation, Canonical Correlation Analysis (CCA),
# and Linear Regression
# Dataset : DailyDelhiClimateTest.csv
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------
# Step 1 : Load Dataset
# -------------------------------------------------------

file_path = "DailyDelhiClimateTest.csv"

# Google Colab
# file_path="/content/DailyDelhiClimateTest.csv"

try:
    data = pd.read_csv(file_path)
    print("Dataset Loaded Successfully\n")
except FileNotFoundError:
    print("Dataset not found")
    exit()

# -------------------------------------------------------
# Step 2 : Time Series
# -------------------------------------------------------

data["date"] = pd.to_datetime(data["date"])

data.set_index("date", inplace=True)

ts_data = data["meantemp"].ffill()

# -------------------------------------------------------
# Step 3 : Train Test Split
# -------------------------------------------------------

train_size = int(len(ts_data)*0.80)

train = ts_data[:train_size]

test = ts_data[train_size:]

print("Training Samples :",len(train))
print("Testing Samples :",len(test))

# -------------------------------------------------------
# Step 4 : SARIMA Model
# -------------------------------------------------------

model = SARIMAX(
    train,
    order=(1,1,1),
    seasonal_order=(1,1,1,12)
)

model_fit = model.fit()

print(model_fit.summary())

# -------------------------------------------------------
# Step 5 : Forecast
# -------------------------------------------------------

forecast = model_fit.forecast(
    steps=len(test)
)

# -------------------------------------------------------
# Step 6 : Plot Forecast
# -------------------------------------------------------

plt.figure(figsize=(12,6))

plt.plot(train,label="Training")

plt.plot(test,label="Actual")

plt.plot(
    test.index,
    forecast,
    label="SARIMA Forecast"
)

plt.title("SARIMA Forecasting")

plt.xlabel("Date")

plt.ylabel("Mean Temperature")

plt.legend()

plt.grid(True)

plt.show()

# -------------------------------------------------------
# Step 7 : Multivariate Features
# -------------------------------------------------------

features = data[
    [
        "meantemp",
        "humidity",
        "wind_speed"
    ]
]

print("\nCovariance Matrix")
print("="*60)

print(features.cov())

print("\nCorrelation Matrix")
print("="*60)

print(features.corr())

# -------------------------------------------------------
# Step 8 : Canonical Correlation Analysis
# -------------------------------------------------------

X = data[
    [
        "humidity",
        "wind_speed"
    ]
]

Y = data[
    [
        "meantemp"
    ]
]

cca = CCA(n_components=1)

X_c, Y_c = cca.fit_transform(X,Y)

cca_corr = pd.Series(
    X_c[:,0]
).corr(
    pd.Series(Y_c[:,0])
)

print("\nCanonical Correlation")

print("="*60)

print(round(cca_corr,4))

# -------------------------------------------------------
# Step 9 : Linear Regression
# -------------------------------------------------------

reg = LinearRegression()

reg.fit(X,Y)

print("\nRegression Coefficients")

print("="*60)

print(reg.coef_)

print("Intercept")

print(reg.intercept_)

# -------------------------------------------------------
# Step 10 : Completion
# -------------------------------------------------------

print("\nProgram Executed Successfully")
