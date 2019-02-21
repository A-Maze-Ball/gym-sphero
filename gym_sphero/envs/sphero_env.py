import math
import gym
from gym.utils import seeding
import asyncio
import numpy as np

try:
    import spheropy
except ImportError as e:
    raise gym.error.DependencyNotInstalled(f"{e}. (HINT: you can install SpheroPy dependencies with 'pip install spheropy[pygatt]' or pip install spheropy[winble].)")

# Fixed constants
_MIN_SIGNED_16_BIT_INT = -32768
_MAX_SIGNED_16_BIT_INT = 32767

# Globals that can be used for configuration.
# Alternative would be to have a seperate init function.
USE_BLE = True
SPHERO_SEARCH_NAME = 'SK'
NUM_COLLISIONS_TO_RECORD = 5
MIN_COLLISION_THRESHOLD = 60
COLLISION_DEAD_TIME_IN_10MS = 20 # 200 ms
MIN_VELOCITY_MAGNITUDE = 3

class SpheroEnv(gym.Env):
    """
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()

        # Setup gym related members
        # Action is [speed, heading]
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([255, 359]),
            dtype=int
        )

        global USE_BLE
        self._use_ble = USE_BLE

        global SPHERO_SEARCH_NAME
        self._sphero_search_name = SPHERO_SEARCH_NAME

        global NUM_COLLISIONS_TO_RECORD
        self._num_collisions_to_record = NUM_COLLISIONS_TO_RECORD

        global MIN_COLLISION_THRESHOLD
        self._min_collision_threshold = MIN_COLLISION_THRESHOLD

        global COLLISION_DEAD_TIME_IN_10MS
        self._collision_dead_time = COLLISION_DEAD_TIME_IN_10MS

        global MIN_VELOCITY_MAGNITUDE
        self._min_velocity_magnitude = MIN_VELOCITY_MAGNITUDE

        self.observation_space = gym.spaces.Tuple((
            # (x, y) position in cm
            gym.spaces.Box(low=_MIN_SIGNED_16_BIT_INT, high=_MAX_SIGNED_16_BIT_INT,
                shape=(2,), dtype=int),
            # (x, y) velocity in cm/sec
            gym.spaces.Box(low=_MIN_SIGNED_16_BIT_INT, high=_MAX_SIGNED_16_BIT_INT,
                shape=(2,), dtype=int),
            # (x, y) magnitudes of collisions encoutered
            gym.spaces.Box(low=_MIN_SIGNED_16_BIT_INT, high=_MAX_SIGNED_16_BIT_INT,
                shape=(self._num_collisions_to_record, 2), dtype=int),
            # num collisions
            gym.spaces.Box(low=0, high=self._num_collisions_to_record,
                shape=(1,), dtype=int)
        ))

        self.seed()

        # Setup sphero
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._setup_sphero())

    def seed(self, seed=None):
        # Standard seed impl
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        return self.loop.run_until_complete(self.step_async(action))

    async def step_async(self, action):
        debug_info = {}
        # Get observation
        obs = await self._get_obs_async()
        self._reset_collisions()
        reward = self._calc_reward(obs)
        # TODO: Calculate done
        done = False

        # take action
        await self.sphero.roll(action[0], action[1])

        return obs, reward, done, debug_info

    def reset(self):
        return self.loop.run_until_complete(self.reset_async())

    async def reset_async(self):
        await self.sphero.roll(0, 0, mode=spheropy.RollMode.IN_PLACE_ROTATE)
        self._reset_collisions()
        await self.sphero.set_rgb_led(0, 255, 0) # Green
        return await self._get_obs_async()

    def render(self, mode='human', close=False):
        # TODO: Implement some rendering
        pass

    def close(self):
        self.sphero.disconnect()

    async def _setup_sphero(self):
        self.sphero = spheropy.Sphero()

        await self.sphero.connect(
            search_name=self._sphero_search_name,
            use_ble=self._use_ble,
            num_retry_attempts=3
        )

        await self.sphero.set_rgb_led(255, 255, 255) # White
        await self._aim_async()
        await self._configure_collisions_async()

    async def _configure_collisions_async(self):
        self._reset_collisions()
        def handle_collision(data):
            nonlocal self
            if self._num_collisions_since_last_action < self._num_collisions_to_record:
                self._collisions_since_last_action[self._num_collisions_since_last_action] = (data.x_magnitude, data.y_magnitude)
                self._num_collisions_since_last_action += 1

            # fire and forget changing color
            event_loop = asyncio.new_event_loop()
            event_loop.run_until_complete(self._flash_red_async())

        self.sphero.on_collision.append(handle_collision)
        await self.sphero.configure_collision_detection(
            True,
            self._min_collision_threshold, 0,
            self._min_collision_threshold, 0,
            self._collision_dead_time
        )

    async def _aim_async(self):
        original_color = await self.sphero.get_rgb_led()
        input('Place the Sphero at (0, 0) in the environment and press enter:')
        await self.sphero.set_rgb_led()
        await self.sphero.set_back_led(255)
        back_pos = int(input('What is the current back light position in degrees?: '))
        # TODO: Consider allowing to manually rotate the Sphero as well.
        # Would probably need to turn stabilization on and then off again.
        heading_adjustment = ((180 - back_pos) + 360)%360
        await self.sphero.roll(0, heading_adjustment, spheropy.RollMode.IN_PLACE_ROTATE)
        await asyncio.sleep(1) # Give the sphero time to rotate
        await self.sphero.configure_locator()
        await self.sphero.set_heading(0)
        await asyncio.sleep(1)
        await self.sphero.set_back_led(0)
        await self.sphero.set_rgb_led(*original_color)

    async def _get_obs_async(self):
        loc_info = await self.sphero.get_locator_info()
        return (
            np.array([loc_info.pos_x, loc_info.pos_y], dtype=int),
            np.array([loc_info.vel_x, loc_info.vel_y], dtype=int),
            np.array(self._collisions_since_last_action, dtype=int),
            self._num_collisions_since_last_action
        )

    def _reset_collisions(self):
        self._collisions_since_last_action = np.zeros((self._num_collisions_to_record, 2))
        self._num_collisions_since_last_action = 0

    async def _flash_red_async(self):
        await self.sphero.set_rgb_led(255, 0, 0, wait_for_response=False)
        await asyncio.sleep(0.25)
        await self.sphero.set_rgb_led(0, 255, 0)

    def _calc_reward(self, obs):
        vel = obs[1]
        collisions = obs[2]
        vel_norm = np.linalg.norm(vel, ord=2)

        # Negative reward if not moving fast enough.
        if vel_norm < self._min_velocity_magnitude:
            vel_norm = -1

        return round(vel_norm - np.linalg.norm(collisions, ord=2))

