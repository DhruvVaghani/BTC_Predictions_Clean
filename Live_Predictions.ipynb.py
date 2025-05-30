# Databricks notebook source
pip install tensorflow

# COMMAND ----------

# -----------------------------
# Step 1: Imports and Setup
# -----------------------------
from tensorflow.keras.models import load_model
from keras.metrics import MeanSquaredError
import joblib
import numpy as np
import pandas as pd
import pyodbc

# -----------------------------
# Step 2: DB Connection Config
# -----------------------------
jdbcHostname = "dvstock.database.windows.net"
jdbcDatabase = "StockDB"
jdbcPort = 1433
jdbcUsername = "dv"
jdbcPassword = "Kingatbest@123"  # Replace if needed

jdbcUrl = f"jdbc:sqlserver://{jdbcHostname}:{jdbcPort};database={jdbcDatabase};encrypt=true;trustServerCertificate=false;hostNameInCertificate=*.database.windows.net;loginTimeout=30;"

connectionProperties = {
  "user": jdbcUsername,
  "password": jdbcPassword,
  "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}

# -----------------------------
# Step 3: Read Latest Data
# -----------------------------
query = """
(SELECT TOP 10 * 
 FROM StockPrices 
 WHERE symbol = 'btcusd' 
 ORDER BY timestamp DESC) AS recent
"""

df_latest = spark.read.jdbc(url=jdbcUrl, table=query, properties=connectionProperties).toPandas()
df_latest = df_latest.sort_values("timestamp").reset_index(drop=True)

# -----------------------------
# Step 4: Load Model and Scaler
# -----------------------------
model = load_model(
    "/dbfs/FileStore/models/btc_model_v1.h5",
    custom_objects={'mse': MeanSquaredError()}
)

scaler = joblib.load("/dbfs/FileStore/models/btc_scaler.pkl")

# -----------------------------
# Step 5: Make Prediction
# -----------------------------
features = ["open", "high", "low", "close", "volume"]
scaled = scaler.transform(df_latest[features])
X_input = np.expand_dims(scaled, axis=0)  # shape: (1, 10, 5)

y_pred_scaled = model.predict(X_input)

# Inverse transform
pad = np.zeros((1, len(features)))
pad[0, 3] = y_pred_scaled[0, 0]
predicted_close = scaler.inverse_transform(pad)[0, 3]

print(f"🟢 Predicted close for next 5-minute interval: ${predicted_close:.2f}")

# -----------------------------
# Step 6: Insert into SQL Table
# -----------------------------
from datetime import datetime, timedelta

# # Use current time + 5 minutes as the prediction target
# predicted_time = df_latest["timestamp"].iloc[-1] + timedelta(minutes=5)

# try:
#     conn = pyodbc.connect(
#         f"DRIVER={{com.microsoft.sqlserver.jdbc.SQLServerDriver}};"
#         f"SERVER={jdbcHostname};"
#         f"DATABASE={jdbcDatabase};"
#         f"UID={jdbcUsername};"
#         f"PWD={jdbcPassword}"
#     )
    
#     cursor = conn.cursor()
#     cursor.execute("""
#         INSERT INTO BTC_Predictions (timestamp, symbol, predicted_close)
#         VALUES (?, ?, ?)
#     """, (predicted_time, "btcusd", predicted_close))
    
#     conn.commit()
#     cursor.close()
#     conn.close()
#     print("✅ Prediction inserted into PredictedPrices table.")
# except Exception as e:
#     print(f"❌ Failed to insert prediction: {e}")


from datetime import datetime
import numpy as np
import pandas as pd

# -----------------------------
# Step 4: Predict + inverse
# -----------------------------
y_pred_scaled = model.predict(X_input)

# Create a padded array to inverse only 'close'
pad = np.zeros((1, len(features)))
pad[0, 3] = y_pred_scaled[0, 0]  # 3 is index of 'close'
predicted_close = scaler.inverse_transform(pad)[0, 3]

# -----------------------------
# Step 5: Prepare Row for SQL
# -----------------------------
from datetime import datetime, timedelta
prediction_time = (datetime.utcnow() + timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')

pred_df = pd.DataFrame([{
    "prediction_time": prediction_time,
    "predicted_close": float(predicted_close)
}])

# -----------------------------
# Step 6: Write to SQL using Spark
# -----------------------------
# Convert pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(pred_df)

# Correct driver
connectionProperties = {
    "user" : jdbcUsername,
    "password" : jdbcPassword,
    "driver" : "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}

# Append to BTC_Predictions table
spark_df.write.jdbc(
    url=jdbcUrl,
    table="BTC_Predictions",
    mode="append",
    properties=connectionProperties
)

print(f"✅ Prediction inserted: {prediction_time} — ${predicted_close:.2f}")

