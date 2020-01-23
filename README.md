# Deep-Q-Network-pytorch

Pytorch Implementation of Deep Q network Algorithm to solve Banana Navigation Problem (Unity Enviroment).


### Enviroment

In this it will be train an agent to collect Yellow Bananas in a large Square world. 
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. 
Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. 
Given this information, the agent has to learn how to best select actions. 

Four discrete actions are available, corresponding to:

- 0 - move forward.
- 1 - move backward.
- 2 - turn left.
- 3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
2. Place the file in this repository and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  

this README is base on [Udacity Repository](https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/README.md)
