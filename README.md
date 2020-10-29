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

To run the program:

```
python main.py
```

There are three 'modes' available. The training mode trains the agent in the environment for either the specified number of episodes or until the agent has solved the environment, depending on which happens sooner. 

The testing mode renders several episodes of the agent in the environment, a representation of how the agent is performing and whether it solves the environment. There is a load_weights command line parameter than can be used to load trained weights and see how the agent performs.

The playing mode allows the user to play using keyboard input (using 'w' and 's'). 

```
python main.py --mode train
python main.py --mode test
python main.py --mode play
```

There are also various command line arguments that can be used to feed in values for the hyperparameters such as 'lrate' for the learning rate or 'num_episodes' for the number of episodes. Example:

```
python main.py --lrate 7e-4 --num_episodes 1000
```

# Implementation

We use policy gradient learning (based on Andrej Karpathy's blog post and this code: https://github.com/pytorch/ignite/blob/master/examples/reinforcement_learning/reinforce.py) to solve the Pong environment. There is an argument called "players" that can be used to specify how many agents we would like to control when we initialize the environment. 
