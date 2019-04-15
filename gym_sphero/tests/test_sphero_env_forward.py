import gym
import gym_sphero

# TODO: Replace prints with env.render when we implement it.

def main():
    env = gym.make('Sphero-v0')
    env.configure(max_num_steps_in_episode=40, stop_episode_at_collision=True, center_sphero_every_reset=True)

    for _ in range(2):
        state_t = env.reset()
        print(f'Initial State: {state_t}')
        total_reward = 0
        action_t = [40, 0]
        done_t = False
        step_t = 0
        while not done_t:
            state_t, reward_t, done_t, _ = env.step(action_t)
            total_reward += reward_t
            print(f'Action: {action_t}')
            print(f'Previous observertion: {state_t}')
            step_t += 1

        print(f'Total number of steps: {step_t}')
        print(f'Total Reward: {total_reward}')

    env.close()


if __name__ == '__main__':
    main()