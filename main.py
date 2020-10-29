import gym
import numpy as np 
import os
import sys
import time
from argparse import ArgumentParser
from agent import Agent
# from utils import plotLearning

# majority of code obtained from https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code

# ------------------------- Code for keyboard agent ------------------------- #
# from https://github.com/openai/gym/blob/master/examples/agents/keyboard_agent.py

# Keyboard controls:
# w - Nop
# a - fire right engine
# s - fire main engine
# d - fire left engine

do_user_action = False
user_action = -1
main_engine = 0.0
left_right_engine = 0.0 

def key_press(k, mod):
    global do_user_action, user_action
    if k == ord('w'):
        user_action = 0
        do_user_action = True
        main_engine = 0.0
        left_right_engine = 0.0 
    if k == ord('a'):
        user_action = 3
        do_user_action = True
        main_engine = 0.0
        left_right_engine = 0.5

    if k == ord('s'):
        user_action = 2
        do_user_action = True
        main_engine = 0.5
        left_right_engine = 0.0

    if k == ord('d'):
        user_action = 1
        do_user_action = True
        main_engine = 0.0
        left_right_engine = -0.5

def key_release(k, mod):
    global do_user_action, user_action
    do_user_action = False
    user_action = -1
    main_engine = 0.0
    left_engine = 0.0 
    right_engine = 0.0 


# -------------------------------------------------------------------------- #

def main(args):
    env = gym.make('LunarLanderContinuous-v2')

    # Agent initialization
    agent = Agent(alpha=args.actor_lrate, beta=args.critic_lrate, input_dims=[8], tau=args.tau, env=env, 
                batch_size=args.batch_size, layer1_size=400, layer2_size=300, n_actions=2)

    np.random.seed(0)
    score_history = []

    # for training
    if args.mode == "train":
        for i in range(args.num_episodes):
            done = False
            score = 0
            obs = env.reset()
            while not done:
                act = agent.choose_action(obs)
                new_state, reward, done, info = env.step(act)
                agent.remember(obs, act, reward, new_state, int(done))
                agent.learn()
                score += reward
                obs = new_state
            
            score_history.append(score)
            print('episode ', i, 'score %.2f' % score, '100 game average %.2f' % np.mean(score_history[-100:]))

            if i % 25 == 0:
                agent.save_models()

    # for testing
    elif args.mode == "test":
        if args.load_weights:
            agent.load_models()
        
        for i in range(args.num_episodes):
            done = False
            score = 0
            obs = env.reset()
            while not done:
                act = agent.choose_action(obs)
                new_state, reward, done, info = env.step(act)
                score += reward
                obs = new_state
            
            score_history.append(score)
            print('episode ', i, 'score %.2f' % score, '100 game average %.2f' % np.mean(score_history[-100:]))

    # for playing with keyboard
    elif args.mode == "play":
        # enable key presses
        env.render()
        env.unwrapped.viewer.window.on_key_press = key_press
        env.unwrapped.viewer.window.on_key_release = key_release
        global do_user_action, user_action, main_engine, left_right_engine
        state = env.reset()
        total_reward = 0

        prev_action = None
        keypress_cntr = 0.0

        while True:

            my_action = [0.0, 0.0]
            if do_user_action:
                
                # "acceleration" logic
                # the more number of time steps a key remains pressed, the power of engine is increased
                if user_action == 0:
                    my_action[0] = 0.0
                    my_action[1] = 0.0

                    prev_action = 0
                    keypress_cntr = 0.0
                
                elif user_action == 3:
                    my_action[0] = 0.0
                    my_action[1] = 0.5

                    if prev_action == 3:
                        keypress_cntr += 0.1
                    
                    else:
                        prev_action = 3
                        keypress_cntr = 0.0
                    
                    my_action[1] += keypress_cntr

                    if my_action[1] > 1.0:
                        my_action[1] = 1.0

                elif user_action == 2:
                    my_action[0] = 0.5
                    my_action[1] = 0.0

                    if prev_action == 2:
                        keypress_cntr += 0.1
                    
                    else:
                        prev_action = 2
                        keypress_cntr = 0.0
                    
                    my_action[0] += keypress_cntr

                    if my_action[0] > 1.0:
                        my_action[0] = 1.0

                elif user_action == 1:
                    my_action[0] = 0.0
                    my_action[1] = -0.5

                    if prev_action == 1:
                        keypress_cntr += -0.1
                    
                    else:
                        prev_action = 1
                        keypress_cntr = 0.0
                    
                    my_action[1] += keypress_cntr

                    if my_action[1] < -1.0:
                        my_action[1] = -1.0

                else:
                    my_action[0] = 0.0
                    my_action[1] = 0.0

                    prev_action = 0
                    keypress_cntr = 0.0
                
            print(my_action)

            next_state, reward, done, info = env.step(my_action)

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
    parser = ArgumentParser(description='LunarLander-v2 Continuous Agent')

    parser.add_argument('--mode', type=str, default = "test",
                        help="agent mode, ['train', 'test', 'play']")
    parser.add_argument('--num_episodes', type=int, default = 1000,
                        help='number of episodes for training')
    parser.add_argument('--actor_lrate', type=float, default = 0.000025,
                        help='Learning rate')
    parser.add_argument('--critic_lrate', type=float, default = 0.00025,
                        help='Learning rate')
    parser.add_argument('--tau', type=float, default = 0.001,
                        help='Value for tau')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Size of batches for training')
    parser.add_argument('--load_weights', type=bool, default = False,
                        help='load saved weights')

    args = parser.parse_args()
    main(args)






