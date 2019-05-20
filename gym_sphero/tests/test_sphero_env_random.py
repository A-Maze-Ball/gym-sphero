import gym
import gym_sphero

# TODO: Replace prints with env.render when we implement it.

def main():
    env = gym.make('Sphero-v0')
    env.configure(max_num_steps_in_episode=100)
    state_t = env.reset()

    total_reward = 0
    done_t = False
    while not done_t:
        action_t = env.action_space.sample()
        state_t, reward_t, done_t, _ = env.step(action_t)
        total_reward += reward_t
        print(f'Action: {action_t}')

    env.close()
    print(f'Total Reward: {total_reward}')


if __name__ == '__main__':
    main()
