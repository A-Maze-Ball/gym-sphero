import gym
import gym_sphero

NUM_EPISODES = 3
RENDER = True


def main():
    env = gym.make('Sphero-v0')
    env.configure(max_num_steps_in_episode=100)

    for episode in range(NUM_EPISODES):
        state_t = env.reset()
        render(env)
        total_reward = 0
        done_t = False
        step_t = 0
        while not done_t:
            action_t = env.action_space.sample()
            state_t, reward_t, done_t, _ = env.step(action_t)
            if state_t[1] != 0:
                print("Collision!!")

            total_reward += reward_t
            step_t += 1
            render(env)
            print(f'Action: {action_t}')

        env.stop()
        print(f'Total number of steps: {step_t}')
        print(f'Total Reward: {total_reward}')

    env.close()


def render(env):
    global RENDER
    if RENDER:
        env.render()


if __name__ == '__main__':
    main()
