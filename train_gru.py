import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("datasets\\bombay.csv")  

features = ["tempC", "humidity", "windspeedKmph", "pressure", "cloudcover"]
data = df[features].values

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    GRU(50, return_sequences=True, input_shape=(10, 5)),
    GRU(50),
    Dense(25, activation='relu'),
    Dense(1)  
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

model.save("gru_model.h5")

print("GRU model trained and saved as gru_model.h5")
