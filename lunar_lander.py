import gym
import numpy as np
from rl_agent import Agent
from decision_rule import DecisionRule
from collections import deque
from IPython import display
import matplotlib.pyplot as plt
import os
import sys
import time

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# hyperparameters obtained from elsewhere

NUM_EPISODES = 2000
MAX_STEPS = 1000
STATE_SIZE = 8
ACTION_SIZE = 4
LEARNING_RATE = 5e-4
LEARNING_PERIOD = 4
MAX_BUFFER_SIZE = 100000
DISCOUNT = 0.999
TAU = 1e-3
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_END = 0.01
BATCH_SIZE = 64
HIDDEN_LAYER_DIM = 64
LOAD_WEIGHTS = True

env = gym.make('LunarLander-v2')
agent = Agent(state_size=STATE_SIZE, action_size=ACTION_SIZE, learning_rate=LEARNING_RATE,
            learning_period=LEARNING_PERIOD, max_buffer_size=MAX_BUFFER_SIZE, discount=DISCOUNT,
            tau=TAU, epsilon=EPSILON, epsilon_decay=EPSILON_DECAY, epsilon_end=EPSILON_END,
            batch_size=BATCH_SIZE)


scores = []
recent_scores = deque(maxlen=150)

def train():
    print("Training...")
    for i in range(NUM_EPISODES):
        state = env.reset()
        episode_score = 0
        done = False

        for j in range(MAX_STEPS):
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

def test():
    if LOAD_WEIGHTS:
        agent.load_weights()
        agent.set_epsilon(EPSILON_END)

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


env.render()


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


# human_agent_action = 0
# human_wants_restart = False
# human_sets_pause = False
#
# def key_press(key, mod):
#     global human_agent_action, human_wants_restart, human_sets_pause
#     print("KEY PRESSED!")
#
#     if key==0xff0d: human_wants_restart = True
#     if key==32: human_sets_pause = not human_sets_pause
#     a = int( key - ord('0') )
#     if a <= 0 or a >= ACTION_SIZE: return
#     human_agent_action = a
#
# def key_release(key, mod):
#     global human_agent_action
#
#     print("KEY RELEASED!")
#     a = int( key - ord('0') )
#     if a <= 0 or a >= ACTION_SIZE: return
#     if human_agent_action == a:
#         human_agent_action = 0

env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

def play(env):
    global do_user_action, user_action
    state = env.reset()
    total_reward = 0
    decision_maker = DecisionRule()

    while True:
        chosen_action = decision_maker.get_action(state, user_action)
        next_state, reward, done, info = env.step(chosen_action)
        if reward != 0:
            print("reward : ", reward)

        total_reward += reward
        env.render()
        if done:
            break
        state = next_state
        time.sleep(0.05)


# def play(env):
#     global human_agent_action, human_wants_restart, human_sets_pause
#     human_wants_restart = False
#     obser = env.reset()
#     skip = 0
#     total_reward = 0
#     total_timesteps = 0
#     decision_maker = DecisionRule()
#     while 1:
#         if not skip:
#             #print("taking action {}".format(human_agent_action))
#             a = human_agent_action
#             total_timesteps += 1
#             skip = 0
#         else:
#             skip -= 1
#
#         a = decision_maker.get_action(obser, a)
#         obser, r, done, info = env.step(a)
#         if r != 0:
#             print("reward %0.3f" % r)
#         total_reward += r
#         window_still_open = env.render()
#         if window_still_open==False: return False
#         if done: break
#         if human_wants_restart: break
#         while human_sets_pause:
#             env.render()
#             # time.sleep(0.1)
#         time.sleep(0.05)
#     print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))

# --------------------------------------------------------------------------- #


# Uncomment the following functions to train, test, or play

# train()
# test()
play(env)

env.close()
