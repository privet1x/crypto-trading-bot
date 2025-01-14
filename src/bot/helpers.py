import numpy as np
import pandas as pd

STOP_LOSS_PERCENTAGE = 0.01  # 1% stop-loss
TAKE_PROFIT_PERCENTAGE = 0.02  # 2% take-profit


def calculate_trade_amount(client, symbol, percentage):
    # Fetch wallet balance for Unified Trading Account
    response = client.get_wallet_balance(accountType="UNIFIED")

    # Extract the total wallet balance
    try:
        account_balance = float(response["result"]["list"][0]["totalWalletBalance"])  # Use totalWalletBalance directly
    except KeyError as e:
        raise ValueError(f"Error accessing wallet balance: {e}")

    # Fetch the latest BTC price
    kline_response = client.get_kline(symbol=symbol, interval="1")

    try:
        # Parse the 'list' key in the response and get the last close price
        last_price = float(kline_response["result"]["list"][-1][4])  # 4th index corresponds to the 'close' price
    except (KeyError, IndexError) as e:
        raise ValueError(f"Error fetching last price from k-line data: {e}")

    # Calculate trade amount as percentage of balance
    trade_amount = (account_balance * percentage) / last_price

    # Debug: Print calculated trade amount
    print(f"Calculated Trade Amount: {trade_amount}")

    return trade_amount


def close_all_positions(client):
    try:
        # Retrieve all positions for the specified category and settleCoin
        response = client.get_positions(category="linear", settleCoin="USDT")
        positions = response.get("result", {}).get("list", [])

        if not positions:
            print("No open positions to close.")
            return

        for position in positions:
            size = float(position["size"])
            symbol = position["symbol"]
            side = "Sell" if position["side"] == "Buy" else "Buy"

            if size > 0:  # Check if there is an open position
                print(f"Closing position for {symbol}: {size} {side}")
                client.place_order(
                    symbol=symbol,
                    side=side,
                    order_type="Market",
                    qty=size,
                    category="linear",
                )

        print("All positions closed successfully.")
    except Exception as e:
        print(f"Error closing positions: {e}")



def fetch_market_data(client, symbol, interval, limit=100):
    """
    Fetch historical K-line (candlestick) data from Bybit.

    Parameters:
        client: Bybit API client.
        symbol: Trading pair symbol, e.g., "BTCUSDT".
        interval: K-line interval in minutes, e.g., "1", "5", "15".
        limit: Number of data points to fetch.

    Returns:
        np.array: Array of closing prices.
    """
    kline = client.get_kline(symbol=symbol, interval=str(interval), limit=limit)
    if "result" in kline and "list" in kline["result"]:
        price_data = np.array([float(candle[4]) for candle in kline["result"]["list"]])
        return price_data
    else:
        raise ValueError(f"Unexpected K-line response structure: {kline}")


def calculate_risk_management_prices(last_price, side):
    """
    Calculate stop-loss and take-profit prices based on the trade side.

    Parameters:
        last_price: The latest price of the asset.
        side: "Buy" or "Sell".

    Returns:
        stop_loss: The stop-loss price.
        take_profit: The take-profit price.
    """
    if side == "Buy":
        stop_loss = last_price * (1 - STOP_LOSS_PERCENTAGE)
        take_profit = last_price * (1 + TAKE_PROFIT_PERCENTAGE)
    else:  # side == "Sell"
        stop_loss = last_price * (1 + STOP_LOSS_PERCENTAGE)
        take_profit = last_price * (1 - TAKE_PROFIT_PERCENTAGE)

    return stop_loss, take_profit


def calculate_indicators(price_data):
    """
    Calculate technical indicators based on price data.

    Parameters:
        price_data (np.array): Array of close prices.

    Returns:
        np.array: Array of calculated technical indicators.
    """
    # Convert price data into a DataFrame
    df = pd.DataFrame(price_data, columns=["close"])

    # Simple Moving Average (SMA)
    df["sma"] = df["close"].rolling(window=14).mean()

    # Relative Strength Index (RSI)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD and MACD Signal
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Average True Range (ATR)
    high_low = df["close"].rolling(window=1).max() - df["close"].rolling(window=1).min()
    high_close = np.abs(df["close"] - df["close"].shift(1))
    low_close = np.abs(df["close"] - df["close"].shift(1))
    true_range = high_low.combine(high_close, max).combine(low_close, max)
    df["atr"] = true_range.rolling(window=14).mean()

    # Bollinger Band Width (BB Width)
    df["bb_upper"] = df["close"].rolling(window=20).mean() + 2 * df["close"].rolling(window=20).std()
    df["bb_lower"] = df["close"].rolling(window=20).mean() - 2 * df["close"].rolling(window=20).std()
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]

    # Exponential Moving Average (EMA)
    df["ema"] = df["close"].ewm(span=14).mean()

    # Replace NaN values with zeros (important for the first few rows)
    df.fillna(0, inplace=True)

    # Return the indicators as a numpy array
    return df[["sma", "rsi", "macd", "macd_signal", "atr", "bb_width", "ema"]].values
