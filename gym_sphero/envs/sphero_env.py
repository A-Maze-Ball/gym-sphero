import asyncio
import threading
import gym
from gym.utils import seeding
import numpy as np

import cv2
import cv2.aruco

try:
    import spheropy
except ImportError as e:
    raise gym.error.DependencyNotInstalled(
        f"{e}. (HINT: you can install SpheroPy dependencies with 'pip install spheropy[pygatt]' or pip install spheropy[winble].)")

# Fixed constants
_MIN_SIGNED_16_BIT_INT = -32768
_MAX_SIGNED_16_BIT_INT = 32767
# Colors in [red, green, blue]
_COLLISION_COLOR = [255, 0, 0]  # Red
_AIM_COLOR = [255, 255, 255]    # White
_DONE_COLOR = [0, 134, 183]     # Blue
_READY_COLOR = [0, 255, 0]      # Green

# Fixed ARUCO settings
_ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()
_ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)


class _CameraFrameProcessThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self._frame = None
        self._is_goal_reached = False
        self._lock = threading.Lock()

        self._cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self._cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # turn the autofocus off
        self._cam.set(3, 1280)   # set the Horizontal resolution
        self._cam.set(4, 720)    # Set the Vertical resolution

    def run(self):
        while self._cam.isOpened():
            with self._lock:
                success, self._frame = self._cam.read()
                if not success:
                    raise RuntimeError('Could not get image from camera')

                # TODO: we probably want to grayscale this first.
                _, ids, _ = cv2.aruco.detectMarkers(
                    self._frame, _ARUCO_DICT, parameters=_ARUCO_PARAMS)

                if ids is None:
                    self._is_goal_reached = True

    @property
    def frame(self):
        with self._lock:
            return self._frame

    @property
    def is_goal_reached(self):
        with self._lock:
            return self._is_goal_reached

    def reset(self):
        with self._lock:
            self._is_goal_reached = False

    def close(self):
        self._cam.release()
        cv2.destroyAllWindows()


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
        self._image_thread = None  # placeholder
        self._collision_occured = False
        self._num_steps = 0
        self._done = False
        self._flash_return_color = None  # placeholder

        # Create asyncio event loop to run async functions.
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def configure(self,
                  use_ble=True,
                  sphero_search_name='SK',
                  level_sphero=False,
                  center_sphero_every_reset=False,
                  max_num_steps_in_episode=200,
                  stop_episode_at_collision=False,
                  min_collision_threshold=60,
                  collision_dead_time_in_10ms=20,  # 200 ms
                  collision_penalty=-1,
                  step_penalty=-1,
                  goal_reward=100):
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
            center_sphero_every_reset (bool):
                If True, every call to reset will walk through the aim routine.
                If False, only the first reset will walk through the aim routine.
            max_num_steps_in_episode (int):
                The max number of steps to take in an episode.
            stop_episode_at_collision (bool):
                If True, stops the episode after the first collision.
                If False, episode will continue after collision.
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
            collision_penalty (int):
                The penalty to apply when a collision occures.
                Should be negative integer in most cases.
            step_penalty (int):
                The penalty to apply every step of the environment.
                Usually non-positive integer in most cases.
            goal_reward (int):
                The reward to receive when goal is reached.
                Usually a positive integer for most cases.
        """
        self._use_ble = use_ble
        self._sphero_search_name = sphero_search_name
        self._level_sphero = level_sphero
        self._center_sphero_every_reset = center_sphero_every_reset
        self._max_num_steps_in_episode = max_num_steps_in_episode
        self._stop_episode_at_collision = stop_episode_at_collision
        self._min_collision_threshold = min_collision_threshold
        self._collision_dead_time_in_10ms = collision_dead_time_in_10ms
        self._collision_penalty = collision_penalty
        self._step_penalty = step_penalty
        self._goal_reward = goal_reward

        # Observation space depends on some dynamic properties.
        self.observation_space = gym.spaces.Tuple((
            # image
            gym.spaces.Box(low=0, high=255, shape=(100, 100, 3), dtype=int),
            # num collisions
            gym.spaces.Box(low=0, high=1, shape=(1,), dtype=int)
        ))

    def seed(self, seed=None):
        # Standard seed impl
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        return self.loop.run_until_complete(self.step_async(action))

    async def step_async(self, action):
        if self._done:
            raise RuntimeError(
                "Cannot step when environment is in done state.")

        # Must get the frame first
        frame_t = self._image_thread.frame

        is_goal_reached = self._image_thread.is_goal_reached
        obs_t = self._get_obs(frame_t)
        self._reset_collisions()
        reward_t = self._calc_reward(is_goal_reached)
        is_max_steps_reached = self._num_steps + 1 >= self._max_num_steps_in_episode
        stop_for_collision = self._stop_episode_at_collision and self._collision_occured
        self._done = is_goal_reached or is_max_steps_reached or stop_for_collision
        debug_info = {}
        if self._done:
            await self._set_color(*_DONE_COLOR)
        else:
            # take action
            await self._sphero.roll(action[0], action[1])
            self._num_steps += 1

        return obs_t, reward_t, self._done, debug_info

    def reset(self):
        return self.loop.run_until_complete(self.reset_async())

    async def reset_async(self):
        self._num_steps = 0
        self._done = False
        if self._sphero is None:
            await self._setup_sphero()
        elif self._center_sphero_every_reset:
            await self._aim_async()

        # TODO: Consider rolling in a random direction to get a new starting position.
        # if center_sphero_every_reset is false.

        await self._sphero.roll(0, 0, mode=spheropy.RollMode.IN_PLACE_ROTATE)
        await asyncio.sleep(1)  # give the Sphero time to rotate.
        self._reset_collisions()
        await self._set_color(*_READY_COLOR)

        if self._image_thread is None:
            self._image_thread = _CameraFrameProcessThread()

        if not self._image_thread.is_alive():
            self._image_thread.start()
            # Wait till we get the first frame.
            while self._image_thread.frame is None:
                await asyncio.sleep(1)

        self._image_thread.reset()
        frame = self._image_thread.frame
        return self._get_obs(frame)

    def stop(self):
        return self.loop.run_until_complete(self.stop_async())

    async def stop_async(self):
        await self._sphero.roll(0, 0)

    def render(self, mode='human', close=False):
        # TODO: Implement some rendering
        pass

    def close(self):
        try:
            self.stop()
        except:
            pass

        pending = asyncio.all_tasks(loop=self.loop)
        self.loop.run_until_complete(asyncio.gather(*pending))
        self._sphero.disconnect()
        self._sphero = None
        self.loop.close()
        self._image_thread.close()

    async def _setup_sphero(self):
        self._sphero = spheropy.Sphero()

        await self._sphero.connect(
            search_name=self._sphero_search_name,
            use_ble=self._use_ble,
            num_retry_attempts=3
        )

        await self._aim_async()
        await self._configure_collisions_async()

    async def _configure_collisions_async(self):
        self._reset_collisions()

        def handle_collision(data):
            nonlocal self
            self._collision_occured = True

            # fire and forget changing color
            event_loop = asyncio.new_event_loop()
            event_loop.run_until_complete(self._flash_collision_color_async())
            event_loop.close()

        self._sphero.on_collision.append(handle_collision)
        await self._sphero.configure_collision_detection(
            True,
            self._min_collision_threshold, 0,
            self._min_collision_threshold, 0,
            self._collision_dead_time_in_10ms
        )

    async def _aim_async(self):
        await self._set_color(*_AIM_COLOR)

        input('Place the Sphero at (0, 0) in the environment and press enter:')

        await self._set_color()
        await self._sphero.set_back_led(255)
        await self._sphero.set_stabilization(False)

        input('Rotate the Sphero until the back light is at the back and press enter:')

        await self._sphero.set_heading(0)
        await self._sphero.set_stabilization(True)

        await self._sphero.configure_locator()
        await self._sphero.set_heading(0)
        await asyncio.sleep(1)
        await self._sphero.set_back_led(0)
        await self._set_color(*_AIM_COLOR)

        if self._level_sphero:
            await self._set_color()
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

            await self._set_color(*_AIM_COLOR)
            input("Leveling complete. You can let go of the Sphero now and press enter:")

        await asyncio.sleep(1)

    def _get_obs(self, frame_t):
        scaled_frame = cv2.resize(frame_t, (100, 100))
        return (
            scaled_frame,
            1 if self._collision_occured else 0
        )

    def _reset_collisions(self):
        self._collision_occured = False

    async def _flash_collision_color_async(self):
        if self._sphero is not None:
            await self._sphero.set_rgb_led(*_COLLISION_COLOR, wait_for_response=False)
            await asyncio.sleep(0.25)
            await self._sphero.set_rgb_led(*self._flash_return_color, wait_for_response=False)

    async def _set_color(self, r=0, g=0, b=0):
        self._flash_return_color = [r, g, b]
        await self._sphero.set_rgb_led(r, g, b)

    def _calc_reward(self, is_goal_reached):
        reward = self._goal_reward if is_goal_reached else self._step_penalty
        if self._collision_occured:
            reward += self._collision_penalty
        return reward
