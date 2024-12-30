from gym import Env
from gym.spaces import Box
import numpy as np


class BitcoinEnvWrapper(Env):
    def __init__(self, bitcoin_env):
        self.env = bitcoin_env
        self.action_space = Box(low=-1, high=1, shape=(self.env.action_dim,), dtype=np.float32)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.env.state_dim,), dtype=np.float32
        )

    def reset(self):
        return self.env.reset()

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        # Ensure `info` is a dictionary
        if info is None:
            info = {}
        return state, reward, done, info
