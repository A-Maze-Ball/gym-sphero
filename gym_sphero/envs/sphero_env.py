import math
import gym
from gym.utils import seeding
import asyncio
import numpy as np

try:
    import spheropy
except ImportError as e:
    raise gym.error.DependencyNotInstalled(f"{e}. (HINT: you can install SpheroPy dependencies with 'pip install spheropy[pygatt]' or pip install spheropy[winble].)")

class SpheroEnv(gym.Env):
    """
    """
    metadata = {'render.modes': ['human']}

    NUM_COLLISIONS_TO_RECORD = 3

    def __init__(self):
        super().__init__()
        # Setup gym related members

        # speed, heading
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([255, 359]),
            dtype=int
        )

        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=-100, high=100, shape=(2,), dtype=int),           # (x, y) pos in cm
            gym.spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32),      # (x, y) velocity TBD units and range
            gym.spaces.Box(low=-100, high=100, shape=(SpheroEnv.NUM_COLLISIONS_TO_RECORD, 2), dtype=np.float32)    # (x, y) magnitudes of collisions encoutered
        ))

        self.seed()

        # Setup sphero

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.sphero = spheropy.Sphero()

        # TODO: make use_ble more configurable?
        self.loop.run_until_complete(
            self.sphero.connect(
                search_name='SK',
                use_ble=True,num_retry_attempts=3
            )
        )

        # Set to white
        self.loop.run_until_complete(self.sphero.set_rgb_led(255, 255, 255))

        self.loop.run_until_complete(self._aim_async())

        self.collisions_since_last_action = np.zeros((SpheroEnv.NUM_COLLISIONS_TO_RECORD, 2))

    def seed(self, seed=None):
        # Standard impl
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    async def _aim_async(self):
        original_color = await self.sphero.get_rgb_led()
        input('Place the Sphero at (0, 0) in the environment and press enter:')
        await self.sphero.set_rgb_led()
        await self.sphero.set_back_led(255)
        back_pos = int(input('What is the current back light position in degrees?: '))
        heading_adjustment = ((180 - back_pos) + 360)%360
        await self.sphero.set_stabilization(False)
        await self.sphero.set_heading(heading_adjustment)
        await self.sphero.set_stabilization(True)
        await self.sphero.configure_locator()
        await self.sphero.set_back_led(0)
        await self.sphero.set_rgb_led(*original_color)

    def step(self, action):
        return self.loop.run_until_complete(self.step_async(action))

    async def step_async(self, action):
        debug_info = {}
        # Get observation
        obs = await self.sphero.get_locator_info()
        self.collisions_since_last_action = []
        # TODO: calculate reward (or fine tune)
        vel = obs[1]
        collisions = obs[2]
        reward = np.linalg.norm(vel, ord=2) - np.linalg.norm(collisions, ord=2)
        # TODO: Calculate done
        done = False

        # take action
        await self.sphero.roll(action[0], action[1])

        return obs, reward, done, debug_info

    async def _get_obs_async(self):
        loc_info = await self.sphero.get_locator_info()
        return (
            np.array([loc_info.pos_x, loc_info.pos_y], dtype=int),
            np.array([loc_info.vel_x, loc_info.vel_y], dtype=np.float32),
            np.array(self.collisions_since_last_action, dtype=np.float32)
        )

    def reset(self):
        return self.loop.run_until_complete(self.reset_async())

    async def reset_async(self):
        await self.sphero.roll(0, 0)
        self.collisions_since_last_action = []
        await self.sphero.set_rgb_led(0, 255, 0)
        return await self._get_obs_async()

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

