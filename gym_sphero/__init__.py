from gym.envs.registration import register

register(
    id='Sphero-v0',
    entry_point='gym_sphero.envs:SpheroEnv',
    timestep_limit=200, # TODO: choose a meaningful limit.
    reward_threshold=300, # TODO: choose the correct reward threshold
    nondeterministic=True
)