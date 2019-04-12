import asyncio
import threading
import gym
from gym.utils import seeding
import numpy as np

try:
    import spheropy
except ImportError as e:
    raise gym.error.DependencyNotInstalled(
        f"{e}. (HINT: you can install SpheroPy dependencies with 'pip install spheropy[pygatt]' or pip install spheropy[winble].)")

# Fixed constants
_MIN_SIGNED_16_BIT_INT = -32768
_MAX_SIGNED_16_BIT_INT = 32767


class SpheroEnv(gym.Env):
    """
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()

        # Action is [speed, heading]
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([255, 359]),
            dtype=int
        )

        self.configure()
        self.seed()
        self._sphero = None  # placeholder
        self._num_steps = 0

        # Create asyncio event loop to run async functions.
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def configure(self,
                  use_ble=True,
                  sphero_search_name='SK',
                  level_sphero=True,
                  max_num_steps_in_episode=200,
                  num_collisions_to_record=3,
                  min_collision_threshold=60,
                  collision_dead_time_in_10ms=20,  # 200 ms
                  collision_penalty_multiplier=1.0,
                  min_velocity_magnitude=4,
                  low_velocity_penalty=-1,
                  velocity_reward_multiplier=1.0):
        """Configures the environment with the specified values.

        Args:
            use_ble (bool):
                Should BLE be used to connect to the Sphero.
            sphero_search_name (str):
                The partial name to use when
                searching for the Sphero.
            level_sphero (bool):
                If True, the first call to reset will
                try to level the Sphero as part of its
                aim routine.
                If False, leveling will be skipped.
            max_num_steps_in_episode (int):
                The max number of steps to take in an episode.
            num_collisions_to_record (int):
                Number of collisions to include
                in the observation returned from step.
            min_collision_threshold (int):
                Threshold that must be exceeded
                in either x or y direction
                to register a collision.
            collision_dead_time_in_10ms (int):
                The dead time between recording another collision.
                In 10 ms increments so 10 is 100 ms.
            collision_penalty_multiplier (float):
                Multiplier to scale the negative reward
                received when there is a collsion.
                Should be >= 0.
            min_velocity_magnitude (int):
                Minimum velocity that needs to be achieved
                to not incure a penalty.
            low_velocity_penalty (int):
                The penalty to receive when
                min_velocity_magnitude is not achieved.
                Should be <= 0.
            velocity_reward_multiplier (float):
                Multiplier to scale the reward
                received from velocity.
                Should be >= 0.
        """
        self._use_ble = use_ble
        self._sphero_search_name = sphero_search_name
        self._level_sphero = level_sphero
        self._max_num_steps_in_episode = max_num_steps_in_episode
        self._num_collisions_to_record = num_collisions_to_record
        self._min_collision_threshold = min_collision_threshold
        self._collision_dead_time_in_10ms = collision_dead_time_in_10ms
        self._collsion_penalty_multiplier = collision_penalty_multiplier
        self._min_velocity_magnitude = min_velocity_magnitude
        self._low_velocity_penalty = low_velocity_penalty
        self._velocity_reward_multiplier = velocity_reward_multiplier

        # Observation space depends on some dynamic properties.
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

    def seed(self, seed=None):
        # Standard seed impl
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        return self.loop.run_until_complete(self.step_async(action))

    async def step_async(self, action):
        obs = await self._get_obs_async()

        self._reset_collisions()

        reward = self._calc_reward(obs)

        self._num_steps += 1
        done = self._num_steps >= self._max_num_steps_in_episode

        debug_info = {}

        # take action
        await self._sphero.roll(action[0], action[1])

        return obs, reward, done, debug_info

    def reset(self):
        return self.loop.run_until_complete(self.reset_async())

    async def reset_async(self):
        self._num_steps = 0
        if self._sphero is None:
            await self._setup_sphero()

        await self._sphero.roll(0, 0, mode=spheropy.RollMode.IN_PLACE_ROTATE)
        await asyncio.sleep(1)  # give the Sphero time to rotate.
        self._reset_collisions()
        await self._sphero.set_rgb_led(0, 255, 0)  # Green
        return await self._get_obs_async()

    def render(self, mode='human', close=False):
        # TODO: Implement some rendering
        pass

    def close(self):
        self._sphero.disconnect()
        self._sphero = None

    async def _setup_sphero(self):
        self._sphero = spheropy.Sphero()

        await self._sphero.connect(
            search_name=self._sphero_search_name,
            use_ble=self._use_ble,
            num_retry_attempts=3
        )

        await self._sphero.set_rgb_led(255, 255, 255)  # White
        await self._aim_async()
        await self._configure_collisions_async()

    async def _configure_collisions_async(self):
        self._reset_collisions()

        def handle_collision(data):
            nonlocal self
            if self._num_collisions_since_last_action < self._num_collisions_to_record:
                self._collisions_since_last_action[self._num_collisions_since_last_action] = (
                    data.x_magnitude, data.y_magnitude)
                self._num_collisions_since_last_action += 1

            # fire and forget changing color
            event_loop = asyncio.new_event_loop()
            event_loop.run_until_complete(self._flash_red_async())

        self._sphero.on_collision.append(handle_collision)
        await self._sphero.configure_collision_detection(
            True,
            self._min_collision_threshold, 0,
            self._min_collision_threshold, 0,
            self._collision_dead_time_in_10ms
        )

    async def _aim_async(self):
        original_color = await self._sphero.get_rgb_led()

        input('Place the Sphero at (0, 0) in the environment and press enter:')

        await self._sphero.set_rgb_led()
        await self._sphero.set_back_led(255)
        await self._sphero.set_stabilization(False)

        input('Rotate the Sphero until the back light is at the back and press enter:')

        await self._sphero.set_heading(0)
        await self._sphero.set_stabilization(True)

        await self._sphero.configure_locator()
        await self._sphero.set_heading(0)
        await asyncio.sleep(1)
        await self._sphero.set_back_led(0)

        if self._level_sphero:
            level_complete_event = threading.Event()
            level_timeout_in_seconds = 6

            def handle_level_complete(result):
                nonlocal level_complete_event
                level_complete_event.set()

            self._sphero.on_self_level_complete.append(handle_level_complete)

            print("Please hold the Sphero in place while leveling in...")
            print("5")
            await asyncio.sleep(1)
            print("4")
            await asyncio.sleep(1)
            print("3")
            await asyncio.sleep(1)
            print("2")
            await asyncio.sleep(1)
            print("1")
            await asyncio.sleep(1)

            await self._sphero.self_level(timeout=level_timeout_in_seconds)

            level_complete_event.wait(level_timeout_in_seconds)
            level_complete_event.clear()

            await self._sphero.set_rgb_led(*original_color)
            input("Leveling complete. You can let go of the Sphero now and press enter:")

        await asyncio.sleep(1)

        await self._sphero.set_rgb_led(*original_color)

    async def _get_obs_async(self):
        loc_info = await self._sphero.get_locator_info()
        return (
            np.array([loc_info.pos_x, loc_info.pos_y], dtype=int),
            np.array([loc_info.vel_x, loc_info.vel_y], dtype=int),
            np.array(self._collisions_since_last_action, dtype=int),
            self._num_collisions_since_last_action
        )

    def _reset_collisions(self):
        self._collisions_since_last_action = np.zeros(
            (self._num_collisions_to_record, 2))
        self._num_collisions_since_last_action = 0

    async def _flash_red_async(self):
        # Red
        await self._sphero.set_rgb_led(255, 0, 0, wait_for_response=False)
        await asyncio.sleep(0.25)
        await self._sphero.set_rgb_led(0, 255, 0)  # Green

    def _calc_reward(self, obs):
        vel = obs[1]
        collisions = obs[2]
        vel_norm = np.linalg.norm(vel, ord=2)

        if vel_norm < self._min_velocity_magnitude:
            # Negative reward if not moving fast enough.
            vel_reward = self._low_velocity_penalty
        else:
            # Scaled reward based on velocity magnitude.
            vel_reward = vel_norm*self._velocity_reward_multiplier

        # Scaled penalty based on combined collision magnitudes.
        collision_penalty = sum([np.linalg.norm(collision, ord=2)
                                 for collision in collisions])*self._collsion_penalty_multiplier
        return round(vel_reward - collision_penalty)
