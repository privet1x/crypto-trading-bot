import matplotlib.pyplot as plt
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.vec_env import DummyVecEnv
from src.MyBitcoinEnv import MyBitcoinEnv
from src.train_agent import load_processed_data

# Paths to models
MODEL_PATHS = {
    "1m": "../models/drl_agent_1m.zip",
    "5m": "../models/drl_agent_5m.zip",
    "15m": "../models/drl_agent_15m.zip"
}

# Backtesting date ranges
BACKTEST_DATES = {
    "1m": {"start": "2024-12-25", "end": "2024-12-30"},
    "5m": {"start": "2024-11-15", "end": "2024-12-30"},
    "15m": {"start": "2024-11-11", "end": "2024-12-30"}
}

# Data files
DATA_FILES = {
    "1m": "../data/processed_btcusdt_1m.csv",
    "5m": "../data/processed_btcusdt_5m.csv",
    "15m": "../data/processed_btcusdt_15m.csv"
}

# Initial account balance
INITIAL_ACCOUNT = 1e5


def filter_data(df, start_date, end_date):
    """Filter data for the backtesting date range."""
    return df[(df['close_datetime'] >= start_date) & (df['close_datetime'] <= end_date)]


def create_environment(data, initial_account):
    """Create the MyBitcoinEnv environment for backtesting."""
    price_ary = data[["close"]].values
    tech_ary = data[["sma", "rsi", "macd", "macd_signal", "atr", "bb_width", "ema"]].values

    env = MyBitcoinEnv(
        price_ary=price_ary,
        tech_ary=tech_ary,
        initial_account=initial_account,
        mode="test"
    )
    return env


def backtest_model(model_path, env):
    """Run backtesting for a single model."""
    # Load the model
    agent = DRLAgent(env)
    model = agent.get_model(
        model_name="ppo",  # Specify the same model type as used in training
    )
    model.load(model_path)

    # Wrap the environment
    vec_env = DummyVecEnv([lambda: env])

    # Initialize variables for tracking performance
    obs = vec_env.reset()
    portfolio_values = [INITIAL_ACCOUNT]
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, _, done, info = vec_env.step(action)

        # Append portfolio value
        portfolio_values.append(env.total_asset)

    return portfolio_values


def plot_performance(timeframes, results):
    """Generate and save portfolio performance plots."""
    plt.figure(figsize=(12, 6))
    for timeframe, values in results.items():
        plt.plot(values, label=f"{timeframe} Timeframe")

    plt.title("Portfolio Performance During Backtesting")
    plt.xlabel("Time Steps")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid()
    plt.savefig("../results/backtesting_performance.png")
    plt.show()


if __name__ == "__main__":
    # Dictionary to store results for each timeframe
    backtest_results = {}

    for timeframe, model_path in MODEL_PATHS.items():
        print(f"Backtesting model for {timeframe} timeframe...")

        # Load and filter data
        data = load_processed_data(DATA_FILES[timeframe])
        dates = BACKTEST_DATES[timeframe]
        filtered_data = filter_data(data, dates["start"], dates["end"])

        # Create environment
        env = create_environment(filtered_data, INITIAL_ACCOUNT)

        # Run backtesting
        portfolio_values = backtest_model(model_path, env)

        # Store results
        backtest_results[timeframe] = portfolio_values

    # Generate and save performance plot
    plot_performance(MODEL_PATHS.keys(), backtest_results)
