# Ex.No: 6               HOLT WINTERS METHOD
### Date: 19.05.2026



### AIM:
To implement the Holt Winters Method Model using Python.


### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset, perform data exploration
data = pd.read_csv("FINAL_USO.csv")

# Resample and plot data
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)
# Select the Close price column
price_data = data[["Close"]].copy()
price_data.columns = ["Price"]
data_monthly = price_data.resample("MS").mean()  # Month start
print(data_monthly.head())
data_monthly.plot()

# Scale the data and check for seasonality
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),
    index=data_monthly.index,
)
scaled_data.plot()

# Seasonal decomposition
decomposition = seasonal_decompose(data_monthly, model="additive")
decomposition.plot()
plt.show()

# Split test, train data, create a model using Holt-Winters method, train with train data and Evaluate
scaled_data = (
    scaled_data + 1
)  # multiplicative seasonality can't handle non-positive values
train_data = scaled_data[: int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8) :]

model_add = ExponentialSmoothing(
    train_data, trend="add", seasonal="mul", seasonal_periods=12
).fit()
test_predictions_add = model_add.forecast(steps=len(test_data))

ax = train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add", "test_data"])
ax.set_title("Visual evaluation")
plt.show()

print(np.sqrt(mean_squared_error(test_data, test_predictions_add)))
print("RMS Error:", np.sqrt(mean_squared_error(test_data, test_predictions_add)))
print(np.sqrt(scaled_data.var()), scaled_data.mean())

# Create the final model and predict future data and plot it
final_model = ExponentialSmoothing(
    scaled_data, trend="add", seasonal="mul", seasonal_periods=12
).fit()
final_predictions = final_model.forecast(steps=int(len(data_monthly) / 4))

ax = data_monthly.plot()
final_predictions.plot(ax=ax)
ax.legend(["data_monthly", "final_predictions"])
ax.set_xlabel("Date")
ax.set_ylabel("Gold Price (USD)")
ax.set_title("Prediction")
plt.show()
```
### OUTPUT:
<img width="552" height="432" alt="image" src="https://github.com/user-attachments/assets/958c1a5d-f7e1-483c-89e0-c3a3427e92ed" />
<img width="629" height="470" alt="image" src="https://github.com/user-attachments/assets/d6841ecf-1340-45ad-8b8f-ef5ea2c1f1ca" />
<img width="547" height="455" alt="image" src="https://github.com/user-attachments/assets/37434322-dad9-4409-bb83-b54b56137d4a" />
<img width="655" height="100" alt="image" src="https://github.com/user-attachments/assets/4ed387a5-3717-4952-9d39-c4bccca5e6b4" />
<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/5e238e70-d37e-4c42-89c2-932a0ea59ded" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
