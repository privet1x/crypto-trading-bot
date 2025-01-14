import os
import time
from dotenv import load_dotenv
from stable_baselines3 import PPO
from pybit.unified_trading import HTTP
from src.MyBitcoinEnv import MyBitcoinEnv
from helpers import (calculate_trade_amount, close_all_positions, fetch_market_data,
                     calculate_risk_management_prices, calculate_indicators)

load_dotenv()

# Configuration
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# Define model paths
MODEL_PATHS = {
    "1m": "../../models/drl_agent_1m.zip",
    "5m": "../../models/drl_agent_5m.zip",
    "15m": "../../models/drl_agent_15m.zip"
}

# Define timeframes for market data
TIMEFRAMES = {
    "1m": 1,
    "5m": 5,
    "15m": 15
}

# Initialize the Bybit client
client = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=True)

# Risk management configuration

TRADE_AMOUNT = calculate_trade_amount(client,  "BTCUSDT",0.01)  # 1% of your USDT balance
TRADE_AMOUNT = round(TRADE_AMOUNT, 3)
RUN_TIME_SECONDS = 10  # Run for 120sec


def place_trade(client, side, qty, stop_loss=None, take_profit=None):
    """
    Place a market order and optionally specify stop-loss and take-profit levels.

    Parameters:
        client: The Bybit unified trading client.
        side: 'Buy' or 'Sell'.
        qty: Quantity of BTC to trade.
        stop_loss: Optional stop-loss price.
        take_profit: Optional take-profit price.

    Returns:
        The API response for the placed order.
    """
    try:
        # Define order parameters
        order_params = {
            "symbol": "BTCUSDT",
            "side": side,
            "order_type": "Market",
            "qty": qty,
            "category": "linear",  # Use "spot" if trading spot markets.
        }

        # Call the place_order method
        response = client.place_order(callback=None, **order_params)
        print(f"Order Response: {response}")

        # Optionally handle stop-loss and take-profit (not implemented yet)
        if stop_loss or take_profit:
            print(f"Stop Loss: {stop_loss}, Take Profit: {take_profit}")
            # Add conditional order logic here if needed.

        return response

    except Exception as e:
        print(f"Error placing trade: {e}")
        raise



def trade_with_model(model_path, timeframe):
    """Run the trading strategy using a specific model."""
    # Load trained model
    model = PPO.load(model_path)

    # Fetch market data
    market_data = fetch_market_data(client, "BTCUSDT", TIMEFRAMES[timeframe])
    tech_data = calculate_indicators(market_data)

    # Simulate an environment for the model
    env = MyBitcoinEnv(price_ary=market_data.reshape(-1, 1), tech_ary=tech_data, mode="trade")

    # Get the initial state
    obs = env.reset()

    # Use the model to predict actions
    action, _ = model.predict(obs)
    action = action[0]  # Extract the scalar action for this step

    # Determine the trade direction
    side = "Buy" if action > 0 else "Sell"

    # Calculate stop-loss and take-profit prices
    last_price = market_data[-1]
    stop_loss, take_profit = calculate_risk_management_prices(last_price, side)

    # Place the trade with risk management
    place_trade(client=client, side=side, qty=TRADE_AMOUNT, stop_loss=stop_loss, take_profit=take_profit)


if __name__ == "__main__":
    start_time = time.time()
    while True:
        # Run trading for each model and timeframe
        for timeframe, model_path in MODEL_PATHS.items():
            print(f"Running trading bot for {timeframe} timeframe...")
            trade_with_model(model_path=model_path, timeframe=timeframe)

        # Check if the runtime has exceeded the limit
        elapsed_time = time.time() - start_time
        if elapsed_time > RUN_TIME_SECONDS:
            print("Time limit reached. Closing all positions...")
            close_all_positions(client)
            break  # Exit the loop and stop the bot

        # Optional: Add a sleep time between iterations to avoid frequent API calls
        time.sleep(10)  # Wait for 1 minute before the next iteration
