import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TimeSeriesPredictorEnv(gym.Env):
    def __init__(self, data, look_back=10):
        super(TimeSeriesPredictorEnv, self).__init__()
        self.data = data
        self.look_back = look_back
        self.current_step = look_back
        self.done = False

        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(look_back,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

    def step(self, action):
        reward = -abs(action[0] - self.data[self.current_step])

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        return (
            self.data[self.current_step - self.look_back : self.current_step],
            reward,
            self.done,
            {},
        )

    def reset(self):
        self.current_step = self.look_back
        self.done = False
        return self.data[: self.look_back]

    def render(self, mode="human", close=False):
        pass
