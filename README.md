# Sphero OpenAI Gym Environment
Gym environment to control a Sphero robot in real time.

# Install
```
pip install gym_sphero
```

Typically you will also want to install a bluetooth interface for SpheroPy.
For Windows you will typically want:
```
pip install spheropy[winble]
```
For other platforms you will want:
```
pip install spheropy[pygatt]
```

## Raspberry Pi
```
sudo apt-get install pip3 python3-scipy g++ gcc
sudo pip3 install gym_sphero
```
Then you need to install a bluetooth interface for SpheroPy.\
For BLE support:
```
sudo apt-get install gattool
sudo pip3 install spheropy[pygatt]
```
For normal bluetooth support:
```
sudo pip3 install spheropy[pybluez]
```

# Dependencies

- Python 3.6 or greater
- Gym
- NumPy
- SpheroPy
    - Requires a bluetooth interface. See Dependencies and Install at [SpheroPy](https://github.com/irvinec/SpheroPy).

# Gym Environment Details

## Action Space
`Box(2,)`

Data type is int.\
[speed, heading]\
speed in [0, 255]
heading in [0, 359].

## Observation Space
`Tuple( ( Box(2,), Box(2,), Box(C, 2), Box(1,) ) )`

```
(
    [x position in cm, y position in cm],
    [x velocity in cm/sec, y velocity in cm/sec],
    [
        [x_1 collision magnitude, y_1 collision magnitude],
        [x_2 collision magnitude, y_2 collision magnitude],
        ...
        [x_C collision magnitude, y_C collision magnitude]
    ],
    number of collisions
)
```
All data types are ints.

x position and y position in [min 16 bit signed int, max 16 bit signed int]\
x velocity and y velocity in [min 16 bit signed int, max 16 bit signed int]\
For all n in [1, C], x_n collision magnitude and y_n collision magnitude in [min 16 bit signed int, max 16 bit signed int]\
number of collisions in [0, C]\
C is a configurable value. It is the number max number of collisions to record between steps.

## Reward
Data type is int.\
Reward is calculated from a linear combination of velocity and collision magnitudes at each step and rounded to nearest int.\
If velocity is too low, a fixed penalty is given instead of a reward.\
Reward in (-inf, inf).\

## Delayed Observation
Normally with Gym environments, the `step(action)` function returns the observation at time t and the reward at t - 1.\
This environment has a delays observations by 1 time step. So calling `step(action)` returns the observation at t - 1 and reward at t - 1.

# Usage
```
import gym
import gym_sphero

env = gym.make('Sphero-v0')

# Optionally configure the environment with specific values
env.configure(use_ble=True, sphero_search_name='Billy')

# Must reset the environment before using it.
# reset will connect to the Sphero and prompt to configure it the first time it is called.
obs = env.reset()

action = [0, 20]
done = False
total_reward = 0
while not done:
    obs, reward, done, _ = env.step(action)
    total_reward += reward

env.close()
print(f'Total Reward: {total_reward}')
```

## `configure` Parameters
- `use_ble = True`
    - Should BLE be used to connect to the Sphero.
- `sphero_search_name = 'SK'`
    - The partial name to use when searching for the Sphero.
- `level_sphero = True`
    - If True, `reset` will attempt to level the Sphero as part of its aim routine.
    - If False, levelling will be skipped.
- `max_num_steps_in_episode = 200`
    - The max number of steps to take in an episode.
- `num_collisions_to_record = 3`
    - Number of collisions to include in the observation returned from step.
- `min_collision_threshold = 60`
    - Threshold that must be exceeded in either x or y direction to register a collision.
- `collision_dead_time_in_10ms = 20`
    - The dead time between recording another collision in 10 ms increments. e.g. 10 is 100 ms.
- `collision_penalty_multiplier = 1.0`
    - Multiplier to scale the negative reward received when there is a collsion. Should be >= 0.
- `min_velocity_magnitude = 4`
    - Minimum velocity that needs to be achieved to not incure a penalty.
- `low_velocity_penalty = -1`
    - The penalty to receive when min_velocity_magnitude is not achieved. Should be <= 0.
- `velocity_reward_multiplier = 1.0`
    - Multiplier to scale the reward received from velocity. Should be >= 0.

## Async Methods
Along with the usual `step` and `reset` functions, we also provide the async equivalents, `step_async` and `reset_async`.
