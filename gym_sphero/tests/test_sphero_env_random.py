import gym
import gym_sphero

# TODO: Replace prints with env.render when we implement it.

def main():
    env = gym.make('Sphero-v0')
    env.configure(max_num_steps_in_episode=100)
    obs = env.reset()
    print(f'Initial Obs: {obs}')

    total_reward = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        print(f'Action: {action}')
        print(f'Previous observertion: {obs}')

    env.close()
    print(f'Total Reward: {total_reward}')

if __name__ == '__main__':
    main()