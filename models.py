import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        print("QNetwork v23")
       
        self.l1 = nn.Linear(state_size,256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(0.2)
        self.l2 = nn.Linear(256,324)
        self.bn2 = nn.BatchNorm1d(324)       
        self.l3 = nn.Linear(324,512)
        self.bn3 = nn.BatchNorm1d(512)
        self.l4 = nn.Linear(512,action_size)
        
    def forward(self, x):
        """Build a network that maps state -> action values."""
        x = self.dp1(F.relu(self.bn1(self.l1(x))))
        x = F.relu(self.bn2(self.l2(x)))
        x = F.relu(self.bn3(self.l3(x)))
        return F.relu(self.l4(x))

    
    
class QNetwork2(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork2, self).__init__()
        self.seed = torch.manual_seed(seed)
        print("QNetwork 150 points")
        self.l1 = nn.Linear(state_size,256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(0.2)
        self.l2 = nn.Linear(256,324)
        self.bn2 = nn.BatchNorm1d(324)       
        self.l3 = nn.Linear(324,512)
        self.bn3 = nn.BatchNorm1d(512)
        self.l4 = nn.Linear(512,128)
        self.bn4 = nn.BatchNorm1d(128)
        self.l5 = nn.Linear(128,action_size)
        
    def forward(self, x):
        """Build a network that maps state -> action values."""
        x = self.dp1(F.relu(self.bn1(self.l1(x))))
        x = F.relu(self.bn2(self.l2(x)))
        x = F.relu(self.bn3(self.l3(x)))
        x = F.relu(self.bn4(self.l4(x)))
        return F.relu(self.l5(x))
