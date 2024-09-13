# -*- coding: utf-8 -*-
"""
This script predicts future glucose levels trained in one patient and tested ina different one based on historical glucose and insulin data using 
a Long Short-Term Memory (LSTM) neural network. The goal is to predict a patient's glucose levels 
several hours into the future by training a model on the glucose and insulin data of another patient.

Created on Wed Mar  2 17:44:23 2022
@author: juanz
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import math
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam

# Read glucose and insulin data for the first patient
df_glucose = pd.read_csv("C:/Users/juanz/OneDrive/Escritorio/TFM/D1NAMO/diabetes_subset/001/glucose.csv")
df_glucose = df_glucose.drop(columns=['type', 'comments'])

df_insulin = pd.read_csv("C:/Users/juanz/OneDrive/Escritorio/TFM/D1NAMO/diabetes_subset/001/insulin.csv")
df_insulin = df_insulin.drop(columns='comment')
df_insulin['slow_insulin'] = df_insulin['slow_insulin'].fillna(0)  # Fill NaNs in slow insulin

# Merge glucose and insulin data into one DataFrame
df_completo = pd.merge(df_insulin, df_glucose, on=['date', 'time'], how='outer')

# Fill NaN values in insulin columns with 0
df_completo['slow_insulin'] = df_completo['slow_insulin'].fillna(0)
df_completo['fast_insulin'] = df_completo['fast_insulin'].fillna(0)
df_completo = df_completo.iloc[2:]  # Skip the first 2 rows if needed

# Combine date and time columns into a single datetime column
df_completo['date'] = df_completo['date'].astype(str)
df_completo['time'] = df_completo['time'].astype(str)
df_completo['datetime'] = pd.to_datetime(df_completo['date'] + ' ' + df_completo['time'])

# Fill missing glucose values using linear interpolation from previous and next values
for i in range(2, len(df_completo) - 1):
    if pd.isna(df_completo.loc[i, 'glucose']):
        prev_val = df_completo.loc[i - 1, 'glucose']
        next_val = df_completo.loc[i + 1, 'glucose']
        if pd.notna(prev_val) and pd.notna(next_val):
            df_completo.loc[i, 'glucose'] = (prev_val + next_val) / 2

# Read glucose and insulin data for the second patient (for testing/predictions)
df_glucose2 = pd.read_csv("C:/Users/juanz/OneDrive/Escritorio/TFM/D1NAMO/diabetes_subset/006/glucose.csv")
df_glucose2 = df_glucose2.drop(columns=['type', 'comments'])

df_insulin2 = pd.read_csv("C:/Users/juanz/OneDrive/Escritorio/TFM/D1NAMO/diabetes_subset/006/insulin.csv")
df_insulin2 = df_insulin2.drop(columns='comment')
df_insulin2['slow_insulin'] = df_insulin2['slow_insulin'].fillna(0)

# Merge glucose and insulin data for the second patient
df_completo2 = pd.merge(df_insulin2, df_glucose2, on=['date', 'time'], how='outer')

# Fill NaN values in the second patient's data
df_completo2['slow_insulin'] = df_completo2['slow_insulin'].fillna(0)
df_completo2['fast_insulin'] = df_completo2['fast_insulin'].fillna(0)
df_completo2 = df_completo2.iloc[2:]

# Combine date and time into a single datetime column
df_completo2['date'] = df_completo2['date'].astype(str)
df_completo2['time'] = df_completo2['time'].astype(str)
df_completo2['datetime'] = pd.to_datetime(df_completo2['date'] + ' ' + df_completo2['time'])

# Fill missing glucose values in the second patient's data using linear interpolation
for i in range(2, len(df_completo2) - 1):
    if pd.isna(df_completo2.loc[i, 'glucose']):
        prev_val = df_completo2.loc[i - 1, 'glucose']
        next_val = df_completo2.loc[i + 1, 'glucose']
        if pd.notna(prev_val) and pd.notna(next_val):
            df_completo2.loc[i, 'glucose'] = (prev_val + next_val) / 2

# Split the first patient's data for training, and use the second patient's data for testing
train_size = int(len(df_completo) * 1)
train_df = df_completo[:train_size]
test_df = df_completo2

# Select glucose and fast insulin columns for training
df_for_training = train_df[["glucose", "fast_insulin"]]

# Normalize the dataset as LSTM networks are sensitive to magnitude
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

# Empty lists to store the reshaped training data
trainX, trainY = [], []

# LSTM requires reshaped input in n_samples x timesteps x n_features format
# Define the number of past samples used for prediction (n_past) and number of future samples predicted (n_future)
n_future = 12  # Number of future glucose samples to predict
n_past = 10    # Number of past samples used for prediction

# Reformat input data into the required shape (n_samples, timesteps, features)
for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.4))
model.add(Dense(trainY.shape[1]))  # Output layer
model.compile(optimizer=Adam(learning_rate=0.0002), loss='mse')
model.summary()

# Train the model and plot the loss
history = model.fit(trainX, trainY, epochs=50, batch_size=16, validation_split=0.1, verbose=1)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

# Normalize the test data
test_data_scaled = scaler.transform(test_df[["glucose", "fast_insulin"]])

# Prepare test data for predictions
testX, testY = [], []
for i in range(n_past, len(test_data_scaled) - n_future + 1):
    testX.append(test_data_scaled[i - n_past:i, :])
    testY.append(test_data_scaled[i + n_future - 1, 0])
testX, testY = np.array(testX), np.array(testY)

# Make predictions on test and training sets
test_pred = model.predict(testX)
train_pred = model.predict(trainX)

# Expand predictions for inverse transformation
train_pred_expanded = np.hstack([train_pred.reshape(-1, 1), np.zeros((train_pred.shape[0], 1))])
test_pred_expanded = np.hstack([test_pred.reshape(-1, 1), np.zeros((test_pred.shape[0], 1))])

# Reverse scaling to get original glucose values
train_pred_original = scaler.inverse_transform(train_pred_expanded)[:, 0]
test_pred_original = scaler.inverse_transform(test_pred_expanded)[:, 0]

# Expand actual data for inverse transformation
trainY_expanded = np.hstack([trainY.reshape(-1, 1), np.zeros((trainY.shape[0], 1))])
testY_expanded = np.hstack([testY.reshape(-1, 1), np.zeros((testY.shape[0], 1))])

# Reverse scaling for actual glucose values
trainY_original = scaler.inverse_transform(trainY_expanded)[:, 0]
testY_original = scaler.inverse_transform(testY_expanded)[:, 0]

# Calculate RMSE for train and test data
train_rmse = math.sqrt(mean_squared_error(trainY_original, train_pred_original))
print("Training RMSE:", train_rmse)

test_rmse = math.sqrt(mean_squared_error(testY_original, test_pred_original))
print("Test RMSE:", test_rmse)

# Plot actual glucose values vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(df_completo2.index, df_completo2['glucose'], label='Actual Glucose', color='blue')
plt.plot(test_df.index[n_past + n_future - 1:], test_pred_original, label='Predicted Glucose', color='red')
plt.title('Actual vs Predicted Glucose')
plt.xlabel('Sample')
plt.ylabel('Glucose Level (cg/dL)')
plt.legend()
plt.grid(False)
plt.show()

# Plot a subset of the data for more detailed comparison
start_sample = 250
end_sample = 650
test_pred_original_aligned = pd.Series(test_pred_original, index=test_df.index[n_past + n_future - 1:])
subset_real_values = df_completo2['glucose'].iloc[start_sample:end_sample]
subset_pred_values = test_pred_original_aligned.iloc[start_sample:end_sample]

plt.figure(figsize=(12, 6))
plt.plot(subset_real_values.reset_index(drop=True), label='Real Values (Samples 1000-1400)', color='blue')
plt.plot(subset_pred_values.reset_index(drop=True), label='Predicted Values (Samples 1000-1400)', color='red')
plt.title('Comparison of Real and Predicted Glucose Levels (Samples 1000-1400)')
plt.ylabel('Glucose Level (cg/dL)')
plt.xlabel('Sample')
plt.legend()
plt.show()
