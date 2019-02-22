from gym.envs.registration import register
from gym_sphero.envs import SpheroEnv

register(
    id='Sphero-v0',
    entry_point='gym_sphero.envs:SpheroEnv',
    nondeterministic=True
)