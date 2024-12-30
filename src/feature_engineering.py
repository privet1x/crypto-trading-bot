import os
import pandas as pd
import ta  # Technical analysis library for indicators
from sklearn.preprocessing import MinMaxScaler

# Configuration
INPUT_FILES = {
    "1m": "../data/btcusdt_1m.csv",
    "5m": "../data/btcusdt_5m.csv",
    "15m": "../data/btcusdt_15m.csv"
}

SCALE_COLUMNS = ["open", "high", "low", "close", "volume"]
LOOKBACK_PERIOD = {
    "1m": 10,
    "5m": 14,
    "15m": 20
}


def generate_output_file(input_file):
    """
    Generate output file path by appending 'processed_' to the file name.
    :param input_file: Path to the input file.
    :return: Path to the output file.
    """
    directory = os.path.dirname(input_file)
    filename = f"processed_{os.path.basename(input_file)}"
    return os.path.join(directory, filename)


def load_data(file_path):
    """Load dataset from CSV."""
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)


def calculate_indicators(df, timeframe):
    """Calculate essential technical indicators."""
    print(f"Calculating technical indicators for {timeframe} timeframe...")

    # Get the lookback period for the current timeframe
    lookback = LOOKBACK_PERIOD[timeframe]

    # Simple Moving Average (SMA)
    df["sma"] = ta.trend.sma_indicator(df["close"], window=lookback)

    # Relative Strength Index (RSI)
    df["rsi"] = ta.momentum.rsi(df["close"], window=lookback)

    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # Volatility (Average True Range)
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=lookback)

    # Bollinger Bands Width
    bollinger = ta.volatility.BollingerBands(df["close"], window=lookback)
    df["bb_width"] = bollinger.bollinger_hband() - bollinger.bollinger_lband()

    # EMA
    df["ema"] = ta.trend.ema_indicator(df["close"], window=lookback)

    return df


def preprocess_data(df):
    """Normalize and preprocess data."""
    print("Normalizing data...")
    scaler = MinMaxScaler()
    df[SCALE_COLUMNS] = scaler.fit_transform(df[SCALE_COLUMNS])
    df.dropna(inplace=True)  # Drop rows with NaN values after indicator calculation
    return df


def save_data(df, file_path):
    """Save processed data to CSV."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Processed data saved to {file_path}")


if __name__ == "__main__":
    for timeframe, input_file in INPUT_FILES.items():
        print(f"\nProcessing {timeframe} timeframe data...")

        # Generate output file path dynamically
        output_file = generate_output_file(input_file)

        # Load data
        data = load_data(input_file)

        # Calculate indicators with dynamic lookback period
        data = calculate_indicators(data, timeframe)

        # Preprocess data
        data = preprocess_data(data)

        # Save processed data
        save_data(data, output_file)
