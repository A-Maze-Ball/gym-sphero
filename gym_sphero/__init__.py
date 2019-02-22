from gym.envs.registration import register

register(
    id='Sphero-v0',
    entry_point='gym_sphero.envs:SpheroEnv',
    nondeterministic=True
)