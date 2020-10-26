# Retro Pong-v0 (Discrete)
Implementation of deep Q-learning network in the OpenAI retro gym environment Pong-v0 with a discrete action space.

## Installation

While there is a default Pong-v0 environment in OpenAI gym, it is not easy to use the environment for multiple agents. In other words, only one of the paddles can be controlled by us. The workaround for this problem is using the Pong-v0 environment in the OpenAI Retro Gym library, which is almost identical to the Gym Pong-v0 environment but you have the option to control both agents in the environment. 

The following dependencies need to be installed. 

```
pip install gym
pip install gym-retro
pip install torch torchvision
pip install ipython
pip install matplotlib
pip install argparse
```

You will then need to download the Atari-2600 ROMs via this link: http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html. Unzip this file and then enter the ROMs directory. After, run

```
python -m retro.import .
```

## Run 

1. Go to following part of the code in lunar_lander.py

```
# Uncomment the following functions to train, test, or play

train()
test()
play(env)
```

2. Uncomment train() to train an agent, test() to simulate the agent solving the environment, and play() to play LunarLander with agent assistance.
