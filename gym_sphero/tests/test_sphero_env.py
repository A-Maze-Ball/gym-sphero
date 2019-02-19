import gym
import gym_sphero

def main():
    env = gym.make('Sphero-v0')
    obs = env.reset()
    print(f'Initial Obs: {obs}')

    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        print(f'Action: {action}')
        print(f'Previous observertion: {obs}')




if __name__ == '__main__':
    main()