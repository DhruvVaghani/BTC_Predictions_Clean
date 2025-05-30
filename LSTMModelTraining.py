# Databricks notebook source
jdbcHostname = "dvstock.database.windows.net"
jdbcDatabase = "StockDB"
jdbcPort = 1433
jdbcUsername = "dv"
jdbcPassword = "Kingatbest@123"

jdbcUrl = f"jdbc:sqlserver://{jdbcHostname}:{jdbcPort};database={jdbcDatabase};encrypt=true;trustServerCertificate=false;hostNameInCertificate=*.database.windows.net;loginTimeout=30;"

connectionProperties = {
  "user" : jdbcUsername,
  "password" : jdbcPassword,
  "driver" : "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}

# Load data from Azure SQL table
from pyspark.sql.utils import AnalysisException
retry_attempts = 3
for attempt in range(retry_attempts):
    try:
        df = spark.read.jdbc(url=jdbcUrl, table="StockPrices", properties=connectionProperties)
        break
    except AnalysisException:
        if attempt < retry_attempts - 1:
            time.sleep(5)  # wait before retrying
        else:
            raise


# Display the data
df.display()


# COMMAND ----------

# MAGIC %pip install tensorflow
# MAGIC %pip install scikit-learn matplotlib joblib
# MAGIC

# COMMAND ----------

df = df.toPandas()



# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# 1. Load your data (already pulled from Azure SQL)
df = df[df["symbol"] == "btcusd"]
df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)



# 2. Feature scaling
features = ["open", "high", "low", "close", "volume"]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

# 3. Create time series sequences
def create_sequences(data, seq_len=10):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len][3])  # index 3 = 'close'
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, seq_len=10)

# 4. Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=4, validation_data=(X_test, y_test))

# 6. Predict and inverse transform
y_pred = model.predict(X_test)

# Add dummy columns to inverse transform (except 'close')
pad = np.zeros((len(y_pred), len(features)))
pad[:, 3] = y_pred[:, 0]  # only set 'close'
y_pred_inv = scaler.inverse_transform(pad)[:, 3]

pad[:, 3] = y_test
y_test_inv = scaler.inverse_transform(pad)[:, 3]

# 7. Plot results
plt.figure(figsize=(10,5))
plt.plot(y_test_inv, label="Actual Close")
plt.plot(y_pred_inv, label="Predicted Close")
plt.legend()
plt.title("BTCUSD LSTM Prediction")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.grid()
plt.show()


# COMMAND ----------

# Save the Keras model to local temporary path
model.save("/tmp/btc_model_v1.h5")

# Save the scaler to local temporary path
import joblib
joblib.dump(scaler, "/tmp/btc_scaler.pkl")


# COMMAND ----------

# ✅ Copy to DBFS
dbutils.fs.cp("file:/tmp/btc_model_v1.h5", "dbfs:/FileStore/models/btc_model_v1.h5")
dbutils.fs.cp("file:/tmp/btc_scaler.pkl", "dbfs:/FileStore/models/btc_scaler.pkl")


# COMMAND ----------

display(dbutils.fs.ls("dbfs:/FileStore/models/"))


# COMMAND ----------

from tensorflow.keras.models import load_model
import joblib
import numpy as np
from keras.metrics import MeanSquaredError


query = """
(SELECT TOP 10 * 
 FROM StockPrices 
 WHERE symbol = 'btcusd' 
 ORDER BY timestamp DESC) AS recent
"""

df_latest = spark.read.jdbc(url=jdbcUrl, table=query, properties=connectionProperties).toPandas()
df_latest = df_latest.sort_values("timestamp").reset_index(drop=True)



# Load artifacts
model = load_model(
    "/dbfs/FileStore/models/btc_model_v1.h5",
    custom_objects={'mse': MeanSquaredError()}
)

scaler = joblib.load("/dbfs/FileStore/models/btc_scaler.pkl")

# Features
features = ["open", "high", "low", "close", "volume"]

# Scale + shape
scaled = scaler.transform(df_latest[features])
X_input = np.expand_dims(scaled, axis=0)  # shape = (1, 10, 5)

# Predict
y_pred_scaled = model.predict(X_input)

# Inverse transform
pad = np.zeros((1, len(features)))
pad[0, 3] = y_pred_scaled[0, 0]
predicted_close = scaler.inverse_transform(pad)[0, 3]

print(f"🟢 Predicted close for next 5-minute interval: ${predicted_close:.2f}")


# COMMAND ----------

display(dbutils.fs.ls("dbfs:/FileStore/models/"))


# COMMAND ----------

from tensorflow.keras.models import load_model
import joblib
import numpy as np
from keras.metrics import MeanSquaredError
from datetime import datetime
import pandas as pd

# 1. Read latest 10 rows
query = """
(SELECT TOP 10 * 
 FROM StockPrices 
 WHERE symbol = 'btcusd' 
 ORDER BY timestamp DESC) AS recent
"""
df_latest = spark.read.jdbc(url=jdbcUrl, table=query, properties=connectionProperties).toPandas()
df_latest = df_latest.sort_values("timestamp").reset_index(drop=True)

# 2. Load model & scaler
model = load_model("/dbfs/FileStore/models/btc_model_v1.h5", custom_objects={'mse': MeanSquaredError()})
scaler = joblib.load("/dbfs/FileStore/models/btc_scaler.pkl")

# 3. Prepare input
features = ["open", "high", "low", "close", "volume"]
scaled = scaler.transform(df_latest[features])
X_input = np.expand_dims(scaled, axis=0)

# 4. Predict + inverse
y_pred_scaled = model.predict(X_input)
pad = np.zeros((1, len(features)))
pad[0, 3] = y_pred_scaled[0, 0]
predicted_close = scaler.inverse_transform(pad)[0, 3]

# 5. Create a pandas row
prediction_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
pred_df = pd.DataFrame([{
    "prediction_time": prediction_time,
    "predicted_close": float(predicted_close)
}])

# 6. Write to SQL using Spark
spark_df = spark.createDataFrame(pred_df)

spark_df.write.jdbc(
    url=jdbcUrl,
    table="BTC_Predictions",
    mode="append",
    properties=connectionProperties
)

print(f"✅ Prediction inserted: {prediction_time} — ${predicted_close:.2f}")
