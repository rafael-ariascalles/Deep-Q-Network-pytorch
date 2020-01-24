# Navigation problem - Deep Q-Network with Pytorch

For this problem of navigation it was use the Q-network Algorithm using Expirience Replay, Fixed Target and Exploration Decay.

### Learning Algorith

Recalling the algorithm from the paper [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) we got:

![Q-Network_algorithm](./algo.PNG)

it has been implemented in the Agent class of the [dqn_agentes.py](dqn_agentes.py) script.

in the construction of the class it's been replicated the first 3 lines of the algorithm. initialization of the Newtworks (local for training and target for evaluation) 

```python
    def __init__(self, state_size, action_size, seed):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        self.t_step = 0
```


Arquitecture

Result

How to Improve
