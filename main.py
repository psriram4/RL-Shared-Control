import gym
import retro
import sys
import os
import time
import numpy as np
from collections import deque
from argparse import ArgumentParser
from agent import Agent
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
    if k == ord('s'):
        user_action = 1
        do_user_action = True



def key_release(k, mod):
    global do_user_action, user_action
    print("KEY RELEASED!")
    do_user_action = False
    user_action = -1

# -------------------------------------------------------------------------- #

def main(args):
    # you can modify this players argument to create a multi agent environment
    env = retro.make(game='Pong-Atari2600', players=1)

    # Agent initialization
    agent = Agent()

    # for training
    if args.mode == "train":
        print("Training...")
        running_reward = None
        reward_sum = 0
        for i_episode in range(args.num_episodes):
            state = env.reset()
            for t in range(args.max_steps):
                state = agent.preprocess(state)
                action = agent.act(state).numpy()

                # retro gym Pong has a very strange discrete action space 
                # basic translation logic needed to ensure env gets proper action
                translated_action = [0, 0, 0, 0, 0, 0, 0, 0]

                if action == 1:
                    translated_action[4] = 1
                elif action == 2:
                    translated_action[5] = 1
                else:
                    translated_action[4] = 0
                    translated_action[5] = 0

                state, reward, done, _ = env.step(translated_action)
                reward_sum += reward

                agent.policy.rewards.append(reward)
                if done:
                    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                    print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
                    reward_sum = 0
                    break

                if reward != 0:
                    print('ep %d: game finished, reward: %f' % (i_episode, reward) + ('' if reward == -1 else ' !!!!!!!'))

            if i_episode % args.batch_size == 0:
                print('ep %d: policy network parameters updating...' % (i_episode))
                agent.learn()

            if i_episode % 50 == 0:
                print('ep %d: model saving...' % (i_episode))
                agent.save_weights()


    # for testing
    elif args.mode == "test":
        if args.load_weights:
            agent.load_weights()

        running_reward = None
        reward_sum = 0
        for i_episode in range(args.num_episodes):
            state = env.reset()
            for t in range(args.max_steps):
                state = agent.preprocess(state)
                action = agent.act(state).numpy()

                # retro gym Pong has a very strange discrete action space 
                # basic translation logic needed to ensure env gets proper action
                translated_action = [0, 0, 0, 0, 0, 0, 0, 0]

                if action == 1:
                    translated_action[4] = 1
                elif action == 2:
                    translated_action[5] = 1
                else:
                    translated_action[4] = 0
                    translated_action[5] = 0

                state, reward, done, _ = env.step(translated_action)
                reward_sum += reward_sum

                if done:
                    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                    print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
                    reward_sum = 0
                    break

                if reward != 0:
                    print('ep %d: game finished, reward: %f' % (i_episode, reward) + ('' if reward == -1 else ' !!!!!!!'))

    # for playing with keyboard
    elif args.mode == "play":
        # enable key presses
        env.render()
        env.unwrapped.viewer.window.on_key_press = key_press
        env.unwrapped.viewer.window.on_key_release = key_release
        global do_user_action, user_action

        user_action = -1
        do_user_action = False

        running_reward = None
        reward_sum = 0
        for i_episode in range(args.num_episodes):
            state = env.reset()
            for t in range(args.max_steps):
                
                # translate action based on key presses
                translated_action = [0, 0, 0, 0, 0, 0, 0, 0]

                if user_action != -1:
                    if user_action == 0:
                        translated_action[4] = 1

                    elif user_action == 1:
                        translated_action[5] = 1

                    else:
                        translated_action[4] = 0
                        translated_action[5] = 0 

                state, reward, done, _ = env.step(translated_action)
                reward_sum += reward_sum

                env.render()
                if done:
                    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                    print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
                    reward_sum = 0
                    break

                if reward != 0:
                    print('ep %d: game finished, reward: %f' % (i_episode, reward) + ('' if reward == -1 else ' !!!!!!!'))

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
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='max number of time steps per episode')
    parser.add_argument('--lrate', type=float, default = 1e-4,
                        help='Learning rate')
    parser.add_argument('--discount', type=float, default = 0.999,
                        help='Discount rate')
    parser.add_argument('--batch_size', type=int, default = 10,
                        help='Size of batches for training')
    parser.add_argument('--load_weights', type=bool, default = False,
                        help='load saved weights')

    args = parser.parse_args()
    main(args)

