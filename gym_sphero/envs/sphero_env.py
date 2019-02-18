import gym

try:
    import spheropy
except ImportError as e:
    raise gym.error.DependencyNotInstalled(f"{e}. (HINT: you can install SpheroPy dependencies with 'pip install spheropy[pygatt]' or pip install spheropy[winble].)")

class SpheroEnv(gym.Env):
    """
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Tuple(
            gym.spaces.Discrete(256),   # speed
            gym.spaces.Discrete(360)    # heading
        )

        self.observation_space = gym.spaces.Tuple(
            gym.spaces.Box(low=0, high=100, shape=(2,1),    # (x, y) pos in cm
            gym.spaces.Box(low=-10, high=10, shape=(2,1),   # (x, y) velocity TBD units and range
            gym.spaces.Box(low=-100, high=100, shape=(2,)   # (x, y) magnitudes of collisions encoutered
        )

    def step(self, action):
        pass

    def reset(self, action):
        pass

    def render(self, mode='human', close=False):
        pass

