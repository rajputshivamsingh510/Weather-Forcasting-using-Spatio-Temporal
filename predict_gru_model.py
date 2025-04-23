import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

model = load_model("gru_model.h5")

df = pd.read_csv(r"datasets\bombay.csv")

features = ["tempC", "humidity", "windspeedKmph", "pressure", "cloudcover"]

scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

seq_length = 10
X_test = []

test_data = df[features].values[-(seq_length + 1):]
X_test.append(test_data[:-1])  

X_test = np.array(X_test).reshape((1, seq_length, len(features)))

predicted_temp = model.predict(X_test)

predicted_temp = scaler.inverse_transform(np.concatenate((predicted_temp, np.zeros((1, 4))), axis=1))[:, 0]

print("Predicted Temperature for the next day (GRU):", predicted_temp[0])
