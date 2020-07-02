import gym
import numpy as np
from agent import Agent 
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

NUM_EPISODES = 100
STATE_SIZE = 8
ACTION_SIZE = 4
LEARNING_RATE = 5e-4
LEARNING_PERIOD = 1
MAX_BUFFER_SIZE = 1000000
DISCOUNT = 0.99
TAU = 1e-3
EPSILON = 1.0
EPSILON_DECAY = 0.02
EPSILON_END = 0.01
BATCH_SIZE = 64
HIDDEN_LAYER_DIM = 64

env = gym.make('LunarLander-v2')
agent = Agent(state_size=STATE_SIZE, action_size=ACTION_SIZE, learning_rate=LEARNING_RATE, 
            learning_period=LEARNING_PERIOD, max_buffer_size=MAX_BUFFER_SIZE, discount=DISCOUNT, 
            tau=TAU, epsilon=EPSILON, epsilon_decay=EPSILON_DECAY, epsilon_end=EPSILON_END, 
            batch_size=BATCH_SIZE, hidden_layer_dim=HIDDEN_LAYER_DIM)
scores = []
avg_scores = []

for i in range(NUM_EPISODES):
    state = env.reset()
    episode_score = 0
    done  = False

    while not done:
        chosen_action = agent.act(state)
        next_state, reward, done, info = env.step(chosen_action)
        episode_score += reward
        agent.save_to_memory(state, chosen_action, reward, next_state, done)
        agent.learn()
        state = next_state

    scores.append(episode_score)
    avg_scores.append(np.mean(scores))
    print("episode # : ", i, " , score : ", episode_score)

agent.save_weights()

episode_nums = [i for i in range(NUM_EPISODES)]


plt.figure()
plt.subplot(121)
plt.plot(episode_nums, scores)
plt.ylabel('Score')
plt.xlabel('Episode #')

plt.subplot(122)
plt.plot(episode_nums, avg_scores)
plt.ylabel('Average score')
plt.xlabel('Episode #')

plt.show()

image_name = "agent_statistics_1.png"
plt.savefig(image_name)