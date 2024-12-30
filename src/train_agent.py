import os
import datetime
import pandas as pd
from finrl.meta.env_cryptocurrency_trading.env_btc_ccxt import BitcoinEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from src.env_wrapper import BitcoinEnvWrapper

# Configuration
DATA_FILES = {
    "1m": "../data/processed_btcusdt_1m.csv",
    "5m": "../data/processed_btcusdt_5m.csv",
    "15m": "../data/processed_btcusdt_15m.csv"
}
TIMEFRAME = "15m"  # Choose timeframe: "1m", "5m", or "15m"
MODEL_PATH = f"../models/drl_agent_{TIMEFRAME}.zip"

# Define date ranges
TRAIN_START_DATE = "2024-01-01"
TRAIN_END_DATE = "2024-10-14"
VAL_START_DATE = "2024-12-02"
VAL_END_DATE = "2024-12-03"
TEST_START_DATE = "2024-10-15"
TEST_END_DATE = "2024-12-30"


def load_processed_data(file_path):
    """Load processed data from a CSV file."""
    df = pd.read_csv(file_path)

    # Parse 'close_datetime' explicitly with apply
    try:
        df['close_datetime'] = df['close_datetime'].apply(
            lambda x: datetime.datetime.strptime(x.split('+')[0], "%Y-%m-%d %H:%M:%S.%f")
        )
        df['close_date'] = df['close_datetime'].dt.date  # Extract date
    except Exception as e:
        print(f"Error parsing 'close_datetime': {e}")
        raise

    return df


def split_data(df):
    """Split the data into training, validation, and test sets."""

    train_data = df[(df['close_datetime'] >= TRAIN_START_DATE) & (df['close_datetime'] <= TRAIN_END_DATE)]
    val_data = df[(df['close_datetime'] >= VAL_START_DATE) & (df['close_datetime'] <= VAL_END_DATE)]
    test_data = df[(df['close_datetime'] >= TEST_START_DATE) & (df['close_datetime'] <= TEST_END_DATE)]

    if train_data.empty or val_data.empty or test_data.empty:
        raise ValueError("One of the datasets (train/val/test) is empty. Check your date ranges.")

    return train_data, val_data, test_data


def train_agent(env, model_path):
    """Train a DRLAgent using PPO."""
    wrapped_env = DummyVecEnv([lambda: BitcoinEnvWrapper(env)])  # Wrap with Gym and VecEnv

    agent = DRLAgent(env=wrapped_env)

    model_kwargs = {
        "learning_rate": 0.005,  # Set a higher learning rate
        "batch_size": 128,  # Optionally adjust batch size
    }
    model = agent.get_model("ppo", model_kwargs=model_kwargs)  # Choose PPO

    tb_log_name = "ppo_training"  # Specify the TensorBoard log name
    trained_model = agent.train_model(model=model, total_timesteps=35000, tb_log_name=tb_log_name)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    trained_model.save(model_path)
    return trained_model


def evaluate_agent(env, model_path):
    """Evaluate a trained DRLAgent."""
    # Load the trained model
    model = PPO.load(model_path)

    wrapped_env = DummyVecEnv([lambda: BitcoinEnvWrapper(env)])  # Wrap the environment
    mean_reward, std_reward = evaluate_policy(model, wrapped_env, n_eval_episodes=10)
    print(f"Evaluation results - Mean Reward: {mean_reward}, Std Reward: {std_reward}")
    return mean_reward, std_reward


if __name__ == "__main__":
    # Load data
    df = load_processed_data(DATA_FILES[TIMEFRAME])

    train_data, _, test_data = split_data(df)  # Validation split is skipped

    # Extract price and technical data for training
    train_price_data = train_data[["close"]].values
    train_tech_data = train_data[["sma", "rsi", "macd", "macd_signal", "atr", "bb_width", "ema"]].values

    env_train = BitcoinEnv(
        price_ary=train_price_data,
        tech_ary=train_tech_data,
        mode="train"
    )

    # Train agent
    trained_model = train_agent(env=env_train, model_path=MODEL_PATH)

    # Extract price and technical data for testing
    test_price_data = test_data[["close"]].values
    test_tech_data = test_data[["sma", "rsi", "macd", "macd_signal", "atr", "bb_width", "ema"]].values

    env_test = BitcoinEnv(
        price_ary=test_price_data,
        tech_ary=test_tech_data,
        mode="test",
        start=0,  # Start from the beginning of the test data
        mid1=0,  # This is required for slicing; set it to 0
        mid2=test_price_data.shape[0],  # Use the full range of test data
        end=test_price_data.shape[0]
    )

    evaluate_agent(env=env_test, model_path=MODEL_PATH)

