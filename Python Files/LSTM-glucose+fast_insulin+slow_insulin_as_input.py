# -*- coding: utf-8 -*-
"""
Objective:
    
This code aims to predict future glucose levels using past glucose data and slow and fast insulin injections 
from diabetic patients. The prediction model is based on an LSTM (Long Short-Term Memory) 
neural network, which is trained on a time series dataset to forecast future glucose levels. 

Created on Wed Mar  2 17:44:23 2022
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


# First, we read the CSV files containing glucose levels and insulin injections.
df_glucose = pd.read_csv("C:/Users/juanz/OneDrive/Escritorio/TFM/D1NAMO/diabetes_subset/001/glucose.csv")
df_glucose = df_glucose.drop(columns=['type', 'comments'])

df_insulin = pd.read_csv("C:/Users/juanz/OneDrive/Escritorio/TFM/D1NAMO/diabetes_subset/001/insulin.csv")
df_insulin = df_insulin.drop(columns='comment')
df_insulin['slow_insulin'] = df_insulin['slow_insulin'].fillna(0)  # Replace NaN with 0 for slow insulin

# Merge insulin and glucose data into one DataFrame
df_completo = pd.merge(df_insulin, df_glucose, on=['date', 'time'], how='outer')

# Fill any remaining NaN values with 0
df_completo['slow_insulin'] = df_completo['slow_insulin'].fillna(0)
df_completo['fast_insulin'] = df_completo['fast_insulin'].fillna(0)
df_completo = df_completo.iloc[2:]

# Join date and time columns into one datetime column
df_completo['date'] = df_completo['date'].astype(str)
df_completo['time'] = df_completo['time'].astype(str)
df_completo['datetime'] = pd.to_datetime(df_completo['date'] + ' ' + df_completo['time'])

# Interpolate missing glucose values using the average of previous and next values
for i in range(2, len(df_completo) - 1):
    if pd.isna(df_completo.loc[i, 'glucose']):
        prev_val = df_completo.loc[i - 1, 'glucose']
        next_val = df_completo.loc[i + 1, 'glucose']
        if pd.notna(prev_val) and pd.notna(next_val):
            df_completo.loc[i, 'glucose'] = (prev_val + next_val) / 2

# Save dates for plotting purposes
train_dates = df_completo['datetime']

# Split data into 70% for training and 30% for testing
train_size = int(len(df_completo) * 0.7)
train_df = df_completo[:train_size]
test_df = df_completo[train_size:]

# Extract relevant features for training: glucose, fast insulin, slow insulin
df_for_training = train_df[["glucose", "fast_insulin", "slow_insulin"]]

# LSTM uses sigmoid and tanh, which are sensitive to magnitude, so we need to normalize the data
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

# Empty lists to store formatted training data
trainX, trainY = [], []

# For LSTM, reshape input data into n_samples x timesteps x n_features.
n_future = 12   # Number of future glucose samples to predict
n_past = 50     # Number of past samples to use for prediction

# Reformat input data into shape (n_samples x timesteps x n_features)
for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
    trainX.append(df_for_training_scaled[i - n_past:i, :])  # Use all features (glucose, fast insulin, slow insulin)
    trainY.append(df_for_training_scaled[i + n_future - 1, 0])  # Glucose is the target for prediction

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

# LSTM model to predict glucose levels using glucose, fast insulin, and slow insulin as input
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(n_past, 3), return_sequences=True))  # 3 features as input
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))  # Dropout layer to avoid overfitting
model.add(Dense(1))  # Output layer with 1 unit (predicting glucose levels)
model.compile(optimizer='adam', loss='mse')
model.summary()

# Fit the model and plot the training and validation losses
history = model.fit(trainX, trainY, epochs=30, batch_size=16, validation_split=0.1, verbose=1)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

# Scale the test data
test_data_scaled = scaler.transform(test_df[["glucose", "fast_insulin", "slow_insulin"]])

# Prepare test data for the LSTM model
testX, testY = [], []
for i in range(n_past, len(test_data_scaled) - n_future + 1):
    testX.append(test_data_scaled[i - n_past:i, :])  
    testY.append(test_data_scaled[i + n_future - 1, 0])  

testX = np.array(testX)  
testY = np.array(testY)  

# Make predictions on test and train data
test_pred = model.predict(testX)
train_pred = model.predict(trainX)

# Expand predictions to unscale (add two columns of 0s, as scaler expects 3 columns)
train_pred_expanded = np.hstack([train_pred.reshape(-1, 1), np.zeros((train_pred.shape[0], 2))])
test_pred_expanded = np.hstack([test_pred.reshape(-1, 1), np.zeros((test_pred.shape[0], 2))])

# Unscale predictions
train_pred_original = scaler.inverse_transform(train_pred_expanded)[:, 0]  # We only need the first column (glucose)
test_pred_original = scaler.inverse_transform(test_pred_expanded)[:, 0]  # Same for test predictions

# Unscale actual target values (train and test)
trainY_expanded = np.hstack([trainY.reshape(-1, 1), np.zeros((trainY.shape[0], 2))])
testY_expanded = np.hstack([testY.reshape(-1, 1), np.zeros((testY.shape[0], 2))])
trainY_original = scaler.inverse_transform(trainY_expanded)[:, 0]  # Only the first column is needed
testY_original = scaler.inverse_transform(testY_expanded)[:, 0]

# Calculate RMSE for training and test data
train_rmse = math.sqrt(mean_squared_error(trainY_original, train_pred_original))
print("Training RMSE:", train_rmse)

test_rmse = math.sqrt(mean_squared_error(testY_original, test_pred_original))
print("Test RMSE:", test_rmse)

# Concatenate real values (train and test)
a = np.append(trainY_original, testY_original)

# Concatenate predicted values (train and test)
b = np.append(train_pred_original, test_pred_original)

# Plot real and predicted values (Train + Test)
plt.figure(figsize=(12, 6))
plt.plot(a, label='Real Values (Train + Test)', color='blue')
plt.plot(b, label='Predicted Values (Train + Test)', color='orange')
plt.title('Comparison of Real and Predicted Glucose Levels')
plt.ylabel('Glucose Level (cg/dL)')
plt.xlabel('Sample')
plt.legend()

# Set y-axis limits and ticks
plt.ylim(0, 23)
plt.yticks(np.arange(0, 22.6, 2.5))

plt.show()

# Plot specific sample range (1000-1400)
start_sample = 1000
end_sample = 1400

# Subset of real and predicted values
subset_real_values = a[start_sample:end_sample]
subset_pred_values = b[start_sample:end_sample]

# Plot comparison for samples 1000-1400
plt.figure(figsize=(12, 6))
plt.plot(subset_real_values, label='Real Values (Samples 1000-1400)', color='blue')
plt.plot(subset_pred_values, label='Predicted Values (Samples 1000-1400)', color='orange')
plt.title('Comparison of Real and Predicted Glucose Levels (Samples 1000-1400)')
plt.ylabel('Glucose Level (cg/dL)') 
plt.xlabel('Sample') 
plt.legend()
plt.show()
