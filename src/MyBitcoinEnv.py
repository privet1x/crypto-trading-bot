from gym import Env
from gym.spaces import Box
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv


class MyBitcoinEnv(Env):  # Custom Gym environment
    def __init__(
        self,
        data_cwd=None,
        price_ary=None,
        tech_ary=None,
        time_frequency=15,
        start=None,
        mid1=172197,
        mid2=216837,
        end=None,
        initial_account=1e6,
        max_stock=1e2,
        transaction_fee_percent=1e-3,
        mode="train",
        gamma=0.99,
    ):
        super(MyBitcoinEnv, self).__init__()

        self.asset_memory = []
        self.action_memory = []

        # Original attributes
        self.stock_dim = 1
        self.initial_account = initial_account
        self.transaction_fee_percent = transaction_fee_percent
        self.max_stock = 1
        self.gamma = gamma
        self.mode = mode

        # Load data
        self.load_data(
            data_cwd, price_ary, tech_ary, time_frequency, start, mid1, mid2, end
        )

        # Define state_dim based on price and tech data dimensions
        self.state_dim = 1 + 1 + self.price_ary.shape[1] + self.tech_ary.shape[1]
        self.action_dim = 1

        # Create `df` for FinRL compatibility
        tech_columns = ["sma", "rsi", "macd", "macd_signal", "atr", "bb_width", "ema"]
        if self.tech_ary.shape[1] != len(tech_columns):
            tech_columns = [f"tech_{i}" for i in range(self.tech_ary.shape[1])]

        self.df = pd.DataFrame(
            data=np.column_stack((self.price_ary.flatten(), self.tech_ary)),
            columns=["close"] + tech_columns
        )

        # Gym attributes
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # Reset values
        self.reset()

    def reset(self):
        self.day = 0
        self.account = self.initial_account
        self.stocks = 0.0
        self.day_price = self.price_ary[self.day]  # Initialize `day_price`
        self.day_tech = self.tech_ary[self.day]  # Initialize `day_tech` if needed
        self.total_asset = self.account + self.day_price[0] * self.stocks
        self.episode_return = 0.0
        self.gamma_return = 0.0

        state = self._get_state()
        return state

    def step(self, action):
        stock_action = action[0]
        adj = self.day_price[0]
        if stock_action < 0:
            stock_action = max(
                0, min(-1 * stock_action, 0.5 * self.total_asset / adj + self.stocks)
            )
            self.account += adj * stock_action * (1 - self.transaction_fee_percent)
            self.stocks -= stock_action
        elif stock_action > 0:
            max_amount = self.account / adj
            stock_action = min(stock_action, max_amount)
            self.account -= adj * stock_action * (1 + self.transaction_fee_percent)
            self.stocks += stock_action

        self.day += 1
        self.day_price = self.price_ary[self.day]  # Update `day_price` for the next step
        self.day_tech = self.tech_ary[self.day]  # Update `day_tech` if needed
        done = (self.day + 1) == self.price_ary.shape[0]

        next_total_asset = self.account + self.day_price[0] * self.stocks
        reward = (next_total_asset - self.total_asset) * 2**-16
        self.total_asset = next_total_asset
        self.asset_memory.append(self.total_asset)

        self.action_memory.append(action)
        state = self._get_state()
        return state, reward, done, {}

    def _get_state(self):
        """Construct the state vector."""
        normalized_tech = [
            self.day_tech[0] * 2**-1,
            self.day_tech[1] * 2**-15,
            self.day_tech[2] * 2**-15,
            self.day_tech[3] * 2**-6,
            self.day_tech[4] * 2**-6,
            self.day_tech[5] * 2**-15,
            self.day_tech[6] * 2**-15,
        ]
        state = np.hstack(
            (
                self.account * 2**-18,
                self.day_price * 2**-15,
                normalized_tech,
                self.stocks * 2**-4,
            )
        ).astype(np.float32)
        return state

    def load_data(
        self, data_cwd, price_ary, tech_ary, time_frequency, start, mid1, mid2, end
    ):
        if data_cwd is not None:
            try:
                price_ary = np.load(f"{data_cwd}/price_ary.npy")
                tech_ary = np.load(f"{data_cwd}/tech_ary.npy")
            except BaseException:
                raise ValueError("Data files not found!")
        self.price_ary = price_ary
        self.tech_ary = tech_ary

    def get_sb_env(self):
        """
        Wrap the environment in a DummyVecEnv for Stable-Baselines3 compatibility
        and return the environment and its initial observation.
        """
        vec_env = DummyVecEnv([lambda: self])
        return vec_env, vec_env.reset()

    def save_asset_memory(self):
        return self.asset_memory  # Return the stored asset values


    def save_action_memory(self):
        return self.action_memory  # Return the stored actions

