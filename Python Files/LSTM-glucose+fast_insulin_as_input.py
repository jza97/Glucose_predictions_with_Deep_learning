# -*- coding: utf-8 -*- 
"""
Created on Wed Mar  2 17:44:23 2022

Objective:
This code aims to predict future glucose levels using past glucose data and fast insulin injections 
from diabetic patients. The prediction model is based on an LSTM (Long Short-Term Memory) 
neural network, which is trained on a time series dataset to forecast future glucose levels. 

@author: juanz
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import math
from sklearn.metrics import mean_squared_error

# Read glucose and insulin data from CSV files
df_glucose = pd.read_csv("C:/Users/juanz/OneDrive/Escritorio/TFM/D1NAMO/diabetes_subset/001/glucose.csv")
df_glucose = df_glucose.drop(columns=['type', 'comments'])

df_insulin = pd.read_csv("C:/Users/juanz/OneDrive/Escritorio/TFM/D1NAMO/diabetes_subset/001/insulin.csv")
df_insulin = df_insulin.drop(columns='comment')
df_insulin['slow_insulin'] = df_insulin['slow_insulin'].fillna(0)

# Merge glucose and insulin data on 'date' and 'time' columns
df_completo = pd.merge(df_insulin, df_glucose, on=['date', 'time'], how='outer')

# Fill missing values with 0s for insulin data
df_completo['slow_insulin'] = df_completo['slow_insulin'].fillna(0)
df_completo['fast_insulin'] = df_completo['fast_insulin'].fillna(0)

# Drop the first 2 rows (likely due to data preprocessing needs)
df_completo = df_completo.iloc[2:]

# Combine date and time into a single datetime column
df_completo['date'] = df_completo['date'].astype(str)
df_completo['time'] = df_completo['time'].astype(str)
df_completo['datetime'] = pd.to_datetime(df_completo['date'] + ' ' + df_completo['time'])

# Fill missing glucose values with the average of the previous and next available values
for i in range(2, len(df_completo) - 1):
    if pd.isna(df_completo.loc[i, 'glucose']):
        prev_val = df_completo.loc[i - 1, 'glucose']
        next_val = df_completo.loc[i + 1, 'glucose']
        if pd.notna(prev_val) and pd.notna(next_val):
            df_completo.loc[i, 'glucose'] = (prev_val + next_val) / 2
                        
# Separate the datetime column into another dataset for training
train_dates = df_completo['datetime']

# Split data into 70% training and 30% testing
train_size = int(len(df_completo) * 0.7)
train_df = df_completo[:train_size]
test_df = df_completo[train_size:]

# Select glucose as the feature to use for training
df_for_training = train_df[["glucose"]]

# Normalize the data since LSTMs are sensitive to the magnitude of the values
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

# Empty lists to store formatted training data
trainX = []
trainY = []

# LSTM requires input data to be reshaped as (n_samples x timesteps x n_features)
n_future = 12  # Number of glucose samples to predict in the future
n_past = 15  # Number of past samples to use for prediction

# Reformat input data into (n_samples x timesteps x n_features)
for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0])  # Only use the glucose column
    trainY.append(df_for_training_scaled[i + n_future - 1, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

# Reshape trainX to add a dimension, as LSTM expects 3D input
trainX = np.expand_dims(trainX, axis=-1)

# Define the LSTM model using glucose data
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(n_past, 1), return_sequences=True))  # Single feature (glucose)
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

# Fit the model and plot the loss and validation loss during training
history = model.fit(trainX, trainY, epochs=30, batch_size=16, validation_split=0.1, verbose=1)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

# Scale the test data to match the normalized training data
test_data_scaled = scaler.transform(test_df[["glucose"]])

# Prepare the test data for predictions
testX, testY = [], []
for i in range(n_past, len(test_data_scaled) - n_future + 1):
    testX.append(test_data_scaled[i - n_past:i, 0])
    testY.append(test_data_scaled[i + n_future - 1, 0])
testX, testY = np.array(testX), np.array(testY)
testX = np.expand_dims(testX, axis=-1)

# Make predictions on both the test and train data
test_pred = model.predict(testX)
train_pred = model.predict(trainX)

# Undo scaling to revert to original glucose levels for evaluation
test_pred_original = scaler.inverse_transform(test_pred.reshape(-1, 1))
testY_original = scaler.inverse_transform(testY.reshape(-1, 1))

train_pred_original = scaler.inverse_transform(train_pred.reshape(-1, 1))
trainY_original = scaler.inverse_transform(trainY.reshape(-1, 1))

# Calculate RMSE for training and testing data
train_rmse = math.sqrt(mean_squared_error(trainY_original, train_pred_original))
print("Training RMSE:", train_rmse)

test_rmse = math.sqrt(mean_squared_error(testY_original, test_pred_original))
print("Test RMSE:", test_rmse)

# Concatenate real values (train + test) for comparison
a = np.append(trainY_original, testY_original)

# Concatenate predicted values (train + test) for comparison
b = np.append(train_pred_original, test_pred_original)

# Plot real and predicted glucose levels
plt.figure(figsize=(12, 6))
plt.plot(a, label='Real Values (Train + Test)', color='blue')
plt.plot(b, label='Predicted Values (Train + Test)', color='orange')
plt.title('Comparison of Real and Predicted Glucose Levels')
plt.ylabel('Glucose Level (cg/dL)')
plt.xlabel('Sample')
plt.legend()
plt.show()

# Define the sample range for a subset of real and predicted values
start_sample = 1000
end_sample = 1400

# Get the subset of real and predicted values within the specified range
subset_real_values = a[start_sample:end_sample]
subset_pred_values = b[start_sample:end_sample]

# Plot the comparison for the subset of samples
plt.figure(figsize=(12, 6))
plt.plot(subset_real_values, label='Real Values (Samples 1000-1400)', color='blue')
plt.plot(subset_pred_values, label='Predicted Values (Samples 1000-1400)', color='orange')
plt.title('Comparison of Real and Predicted Glucose Levels (Samples 1000-1400)')
plt.ylabel('Glucose Level (cg/dL)')
plt.xlabel('Sample')
plt.legend()
plt.show()
