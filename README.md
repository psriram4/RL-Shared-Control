# LunarLander-v2 (Discrete)
Implementation of deep Q-learning network in the OpenAI environment LunarLander-v2 with a discrete action space. 

## Installation

The following dependencies need to be installed:

```
pip install gym
pip install Box2D
pip install torch torchvision
pip install ipython
pip install matplotlib
pip install nose
```

## Run 

To run the program:

```
python main.py
```

There are three 'modes' available. The training mode trains the agent in the environment for either the specified number of episodes or until the agent has solved the environment, depending on which happens sooner. 
The testing mode renders several episodes of the agent in the environment, a visual representation of how the agent is performing and whether it solves the environment.
The playing mode allows the user to play using keyboard input (using 'w', 'a', 's', and 'd'). There is a DecisionRule class that can be used to modify the playing mode and how the action is chosen for the environment.

```
python main.py --mode train
python main.py --mode test
python main.py --mode play
```

There are also various command line arguments that can be used to feed in values for the hyperparameters such as 'lrate' for the learning rate or 'num_episodes' for the number of episodes. Example:

```
python main.py --lrate 7e-4 --num_episodes 1000
```

## Implementation

This implementation uses dueling deep Q-networks for the primary and target networks.
