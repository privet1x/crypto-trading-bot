import os
import binance_history as bh

# Configuration
SYMBOL = "BTCUSDT"
TIMEFRAME = "1m"  # Choose: 1m, 5m, 15m, etc.
START_DATE = "2024-12-01"
END_DATE = "2024-12-30"
OUTPUT_FILE = f"../data/{SYMBOL}_{TIMEFRAME}.csv"

def fetch_and_save_data(symbol, timeframe, start_date, end_date, output_file):
    # Fetch historical data
    print(f"Fetching data for {symbol} ({timeframe}) from {start_date} to {end_date}...")
    data = bh.fetch_klines(
        symbol=symbol,
        timeframe=timeframe,
        start=start_date,
        end=end_date,
    )

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save to CSV
    print(f"Saving data to {output_file}...")
    data.to_csv(output_file, index=False)
    print(f"Data saved successfully.")

if __name__ == "__main__":
    fetch_and_save_data(SYMBOL, TIMEFRAME, START_DATE, END_DATE, OUTPUT_FILE)
