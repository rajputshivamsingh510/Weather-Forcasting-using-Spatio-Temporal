import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional 
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv(r"C:\Users\21200\Desktop\akshat codes\8th sem project\datasets\bombay.csv")
print("Columns in dataset:", df.columns)

if "date_time" in df.columns:
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    print(" Date column successfully converted!")
else:
    print(" 'date_time' column not found!")

df.dropna(subset=["date_time"], inplace=True)
df = df.sort_values(by="date_time")

selected_features = ["tempC", "humidity", "windspeedKmph", "pressure", "cloudcover"]

scaler = MinMaxScaler()
df[selected_features] = scaler.fit_transform(df[selected_features])

target_variable = "tempC"

def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(target[i+seq_length])  
    return np.array(X), np.array(y).reshape(-1, 1)  

seq_length = 10
X, y = create_sequences(df[selected_features].values, df[target_variable].values, seq_length)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f" Data prepared for training: {X_train.shape}, {y_train.shape}")

input_layer = Input(shape=(seq_length, len(selected_features)))

lstm_output = Bidirectional(LSTM(64, return_sequences=False))(input_layer) 

output = Dense(1)(lstm_output)  

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer="adam", loss="mse")

print(" Training the LSTM model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

model.save("weather_lstm_model.h5")
print("Model training complete & saved!")

y_pred = model.predict(X_test)

print(f"🛠 DEBUG: y_pred.shape = {y_pred.shape}, Expected = ({len(X_test)}, 1)")

num_samples = min(y_pred.shape[0], y_test.shape[0])

if y_pred.shape[0] == num_samples:
    y_pred = y_pred.reshape(num_samples, 1)
    y_test_actual = y_test[:num_samples].reshape(num_samples, 1)
else:
    print(f" Shape Mismatch! y_pred: {y_pred.shape}, y_test: {y_test.shape}")
    exit()

dummy_features = np.zeros((num_samples, len(selected_features) - 1))

y_pred = scaler.inverse_transform(np.hstack([y_pred, dummy_features]))[:, 0]
y_test_actual = scaler.inverse_transform(np.hstack([y_test_actual, dummy_features]))[:, 0]

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
mae = mean_absolute_error(y_test_actual, y_pred)
r2 = r2_score(y_test_actual, y_pred)

print("\n Model Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f" MAE: {mae:.2f}")
print(f" R² Score: {r2:.2f}")
