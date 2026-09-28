import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
# Load dataset
data = fetch_california_housing()
# Convert to Pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['PRICE'] = data.target # Add target variable
# Display first few rows
df.head()

# Split features and target variable
X = df.drop('PRICE', axis=1)
y = df['PRICE']
# Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize data for better neural network performance

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a Sequential model
model = Sequential([
 Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
 Dropout(0.2),
 Dense(64, activation='relu'),
 Dropout(0.2),
 Dense(32, activation='relu'),
 Dense(1) # Output layer (single neuron for regression)
])
# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# Train the model
history = model.fit(X_train, y_train, epochs=150, batch_size=16, validation_split=0.2)

loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {loss}")
print(f"Test MAE: {mae}")


y_pred = model.predict(X_test)
# Compare actual vs predicted values
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=y_pred.flatten())
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

