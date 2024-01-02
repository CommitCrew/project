import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt

def normalize_reading(reading):
    mean = reading.mean()
    std = reading.std()
    normalized_reading = (reading - mean) / std
    return normalized_reading

def extractID(input_string):
    return input_string.apply(lambda x: int(''.join(filter(str.isdigit, str(x)))))

dataframe = pd.read_csv('dummy_sensor_data.csv')
# TIME STAMP
dataframe['Timestamp'] = pd.to_datetime(dataframe['Timestamp'])
dataframe['NumericTimestamp'] = dataframe['Timestamp'].astype('int64')  # Convert to Unix timestamp in seconds

# MACHINE 
dataframe['NumericMachine_ID'] = extractID(dataframe['Machine_ID'])

# SENSOR 
dataframe['NumericSensor_ID'] = extractID(dataframe['Sensor_ID'])

# NORMALIZING DATA
dataframe["NormalizedReading"] = normalize_reading(dataframe["Reading"])


#dataframe = dataframe.drop('Timestamp', axis=1)  # Drop the original timestamp column
columns_to_drop = ['Timestamp','Machine_ID', 'Sensor_ID']
dataframe.drop(columns=columns_to_drop,inplace=True)
print(dataframe)
# Split the data into training and testing sets
train_size = int(0.8 * len(dataframe))
train, test = dataframe[:train_size], dataframe[train_size:]

# Define features and target variable
X_train, y_train = train.drop('NormalizedReading', axis=1), train['NormalizedReading']
X_test, y_test = test.drop('NormalizedReading', axis=1), test['NormalizedReading']

# Initialize Random Forest model
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# # Train the model
# rf_model.fit(X_train, y_train)

# # Predict on the test set
# y_pred = rf_model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')

# # Plot actual vs. predicted values
# plt.figure(figsize=(12, 6))
# plt.plot(test.index, y_test, label='Actual')
# plt.plot(test.index, y_pred, label='Predicted')
# plt.legend()
# plt.title('Random Forest Predictions for Time Series')
# plt.show()


xgb_model = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators=100, random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred = xgb_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot actual vs. predicted values
plt.figure(figsize=(12, 6))
plt.plot(test.index, y_test, label='Actual')
plt.plot(test.index, y_pred, label='Predicted')
plt.legend()
plt.title('XGBoost Predictions for Time Series')
plt.show()
