import gym
import sys
import os
import time
import numpy as np
from collections import deque
from argparse import ArgumentParser
from agent import Agent
from decision_rule import DecisionRule
from IPython import display
import matplotlib.pyplot as plt


# ------------------------- Code for keyboard agent ------------------------- #
# from https://github.com/openai/gym/blob/master/examples/agents/keyboard_agent.py

do_user_action = False
user_action = -1

def key_press(k, mod):
    global do_user_action, user_action
    print("KEY PRESSED!")
    if k == ord('w'):
        user_action = 0
        do_user_action = True
    if k == ord('a'):
        user_action = 3
        do_user_action = True
    if k == ord('s'):
        user_action = 2
        do_user_action = True
    if k == ord('d'):
        user_action = 1
        do_user_action = True

def key_release(k, mod):
    global do_user_action, user_action
    print("KEY RELEASED!")
    do_user_action = False
    user_action = -1

# -------------------------------------------------------------------------- #

def main(args):
    env = gym.make('LunarLander-v2')

    # TODO: Agent initialization
    agent = Agent(num_episodes=args.num_episodes, learning_rate=args.lrate, max_steps=args.max_steps, discount=args.discount, batch_size=args.batch_size)

    # for training
    if args.mode == "train":
        print("Training...")
        scores = []
        recent_scores = deque(maxlen=150)

        for i in range(args.num_episodes):
            state = env.reset()
            episode_score = 0
            done = False

            for j in range(args.max_steps):
                if done:
                    break

                chosen_action = agent.act(state)
                next_state, reward, done, info = env.step(chosen_action)
                agent.step(state, chosen_action, reward, next_state, done)
                episode_score += reward
                state = next_state

            scores.append(episode_score)
            recent_scores.append(episode_score)
            agent.reduce_epsilon()

            print("Average of last 100 scores: ", np.mean(recent_scores))

            if np.mean(recent_scores) >= 210.0:
                print("Solved.")
                break

        agent.save_weights()

    # for testing
    elif args.mode == "test":
        if args.load_weights:
            agent.load_weights()
            agent.set_epsilon(0.01)

        for i in range(3):
            state = env.reset()
            img = plt.imshow(env.render(mode='rgb_array'))

            for j in range(200):
                action = agent.act(state)
                img.set_data(env.render(mode='rgb_array'))
                plt.axis('off')
                display.display(plt.gcf())
                display.clear_output(wait=True)
                state, reward, done, _ = env.step(action)
                if done:
                    break

    # for playing with keyboard
    elif args.mode == "play":
        # enable key presses
        env.render()
        env.unwrapped.viewer.window.on_key_press = key_press
        env.unwrapped.viewer.window.on_key_release = key_release
        global do_user_action, user_action
        state = env.reset()
        total_reward = 0
        decision_rule = DecisionRule()

        while True:
            chosen_action = decision_rule.get_action(state, user_action)
            next_state, reward, done, info = env.step(chosen_action)
            if reward != 0:
                print("reward : ", reward)

            total_reward += reward
            env.render()
            if done:
                break
            state = next_state
            time.sleep(0.05)

    else:
        print("Invalid mode!")

    env.close()


if __name__ == "__main__":
    parser = ArgumentParser(description='LunarLander-v2 Agent')

    parser.add_argument('--mode', type=str, default = "test",
                        help="agent mode, ['train', 'test', 'play']")
    parser.add_argument('--num_episodes', type=int, default = 2000,
                        help='number of episodes for training')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='max number of time steps per episode')
    parser.add_argument('--lrate', type=float, default = 5e-4,
                        help='Learning rate')
    parser.add_argument('--discount', type=float, default = 0.999,
                        help='Discount rate')
    parser.add_argument('--batch_size', type=int, default = 64,
                        help='Size of batches for training')
    parser.add_argument('--load_weights', type=bool, default = False,
                        help='load saved weights')

    args = parser.parse_args()
    main(args)


