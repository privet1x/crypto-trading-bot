import os
import datetime
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.evaluation import evaluate_policy
from src.MyBitcoinEnv import MyBitcoinEnv

# Configuration
DATA_FILES = {
    "1m": "../data/processed_btcusdt_1m.csv",
    "5m": "../data/processed_btcusdt_5m.csv",
    "15m": "../data/processed_btcusdt_15m.csv"
}
TIMEFRAME = "5m"  # Choose timeframe: "1m", "5m", or "15m"
MODEL_PATH = f"../models/drl_agent_{TIMEFRAME}.zip"

# Define date ranges
TRAIN_START_DATE = "2024-08-01"
TRAIN_END_DATE = "2024-11-14"
VAL_START_DATE = "2024-12-02"
VAL_END_DATE = "2024-12-03"
TEST_START_DATE = "2024-12-25"
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
    """
    Train a DRLAgent using PPO and FinRL's DRLAgent class.

    Parameters:
        env: The environment object wrapped with GymBitcoinEnvWrapper.
        model_path: The file path where the trained model should be saved.

    Returns:
        trained_model: The trained DRL model.
    """
    # Initialize the DRLAgent with the wrapped environment
    agent = DRLAgent(env)

    # Configure the model
    model_kwargs = {
        "learning_rate": 0.005,  # Set a higher learning rate
        "batch_size": 128,  # Optionally adjust batch size
    }
    model = agent.get_model(
        model_name="ppo",  # Using PPO algorithm
        model_kwargs=model_kwargs,  # Pass model configuration
        tensorboard_log="../logs/"  # Path to store TensorBoard logs
    )

    # Train the model
    tb_log_name = "ppo_training"  # Specify TensorBoard log name
    trained_model = DRLAgent.train_model(
        model=model,
        tb_log_name=tb_log_name,
        total_timesteps=35000  # Define the number of timesteps
    )

    # Save the trained model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    trained_model.save(model_path)
    print(f"Model saved at {model_path}")
    return trained_model


def evaluate_agent(env, model_path):
    """
    Evaluate a trained DRLAgent using Stable-Baselines3's evaluate_policy function.

    Parameters:
        env: The environment object wrapped with GymBitcoinEnvWrapper.
        model_path: The file path to the saved model.

    Returns:
        mean_reward: The mean reward over evaluation episodes.
        std_reward: The standard deviation of the reward over evaluation episodes.
    """
    # Load the trained model
    agent = DRLAgent(env)
    model = agent.get_model(
        model_name="ppo",  # Specify the same model type as used in training
    )
    model.load(model_path)  # Load the saved model from the path

    # Wrap the environment in DummyVecEnv if not already wrapped
    if not isinstance(env, DummyVecEnv):
        vec_env = DummyVecEnv([lambda: env])
    else:
        vec_env = env

    # Use evaluate_policy to compute the mean and standard deviation of rewards
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10, render=False)

    print(f"Evaluation results - Mean Reward: {mean_reward}, Std Reward: {std_reward}")
    return mean_reward, std_reward



if __name__ == "__main__":
    # Load data
    df = load_processed_data(DATA_FILES[TIMEFRAME])

    # Split data into training and testing
    train_data, _, test_data = split_data(df)
    # Validation split is skipped because BitcoinEnv doesnt provide validation mode

    # Extract price and technical data for training
    train_price_data = train_data[["close"]].values
    train_tech_data = train_data[["sma", "rsi", "macd", "macd_signal", "atr", "bb_width", "ema"]].values

    # Initialize training environment
    env_train = MyBitcoinEnv(
        price_ary=train_price_data,
        tech_ary=train_tech_data,
        mode="train"
    )

    # Train agent
    trained_model = train_agent(env=env_train, model_path=MODEL_PATH)

    # Extract price and technical data for testing
    test_price_data = test_data[["close"]].values
    test_tech_data = test_data[["sma", "rsi", "macd", "macd_signal", "atr", "bb_width", "ema"]].values

    # Initialize testing environment
    env_test = MyBitcoinEnv(
        price_ary=test_price_data,
        tech_ary=test_tech_data,
        mode="test",
        start=0,  # Start from the beginning of the test data
        mid1=0,  # This is required for slicing; set it to 0
        mid2=test_price_data.shape[0],  # Use the full range of test data
        end=test_price_data.shape[0]
    )

    # Evaluate the trained model
    evaluate_agent(env=env_test, model_path=MODEL_PATH)



