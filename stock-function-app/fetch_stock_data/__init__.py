import logging
import datetime
import requests
import json
import os
import azure.functions as func
from azure.storage.blob import BlobServiceClient
import pyodbc
from datetime import datetime

def main(mytimer: func.TimerRequest) -> None:
    symbol = "btcusd"  # Example: change if needed
    blob_conn_str = os.getenv("AzureWebJobsStorage")
    container_name = "stockdata"
    TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")

    try:
        url = "https://api.tiingo.com/tiingo/crypto/prices"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {TIINGO_API_KEY}"
        }
        from datetime import datetime, timedelta, timezone

        # Calculate the nearest 5-minute mark (e.g., 04:00, 04:05, ...)
        now_utc = datetime.now(timezone.utc)
        rounded_time = now_utc.replace(minute=(now_utc.minute // 5) * 5, second=0, microsecond=0)

        # Request exactly 1 minute of data
        params = {
            "tickers": symbol,
            "resampleFreq": "1min",
            "startDate": rounded_time.isoformat(),
            "endDate": (rounded_time + timedelta(minutes=1)).isoformat()
        }


        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        logging.info("Step 2: Tiingo API call successful")

        if not data:
            logging.error("API returned empty data.")
            return

        price_entry = data[0]


        price_data_list = price_entry.get("priceData", [])
        if not price_data_list:
            logging.error("❌ priceData list is empty.")
            return

        price_data = price_data_list[0]  # latest 1-min candle
        latest_timestamp = datetime.fromisoformat(price_data["date"].replace("Z", "")).replace(tzinfo=None)

        record = {
            "timestamp": latest_timestamp,
            "symbol": symbol,
            "open": price_data["open"],
            "high": price_data["high"],
            "low": price_data["low"],
            "close": price_data["close"],
            "volume": price_data["volume"]
        }





     # Upload to Azure Blob Storage
        blob_service_client = BlobServiceClient.from_connection_string(blob_conn_str)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=f"{symbol}/{latest_timestamp}.json")
        logging.info("Step 5: Preparing blob upload")
        record["timestamp"] = record["timestamp"].isoformat()
        blob_client.upload_blob(json.dumps(record), overwrite=True)
        logging.info(f"Saved data for {symbol} at {latest_timestamp} to blob storage.")

        # Insert to Azure SQL
        insert_to_sql(record)

    except Exception as e:
        logging.error(f"❌ Function failed: {e}")

def insert_to_sql(record):
    try:
        conn = pyodbc.connect(
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={os.getenv('SQL_SERVER')};"
            f"DATABASE={os.getenv('SQL_DATABASE')};"
            f"UID={os.getenv('SQL_USERNAME')};"
            f"PWD={os.getenv('SQL_PASSWORD')}"
        )

        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO StockPrices (timestamp, symbol, [open], high, low, [close], volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            record["timestamp"],
            record["symbol"],
            record["open"],
            record["high"],
            record["low"],
            record["close"],
            record["volume"]
        ))

        conn.commit()
        cursor.close()
        conn.close()
        logging.info("✅ Data inserted into Azure SQL.")
    
    except Exception as e:
        logging.error(f"❌ SQL Insert failed: {e}")