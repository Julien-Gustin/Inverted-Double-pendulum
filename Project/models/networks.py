import torch.nn as nn 
import torch 
import numpy as np
import random

# Seed
SEED = 42
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class Critic_DDPG(nn.Module):
    def __init__(self, batch:bool, action_space:int, state_space:int, seed:int=42) -> None:
        super(Critic_DDPG, self).__init__()
        torch.manual_seed(seed)

        if batch:
            self.l1 = nn.Sequential(nn.BatchNorm1d(state_space), self.linear_batch_relu(state_space, 400-action_space))

        else:
            self.l1 = nn.Sequential(self.linear_relu(state_space, 400-action_space))

        self.l2 = self.linear_relu(400, 300)
        self.l3 = nn.Linear(300, 1)

        torch.nn.init.uniform_(self.l3.weight, -3*1e-3, 3*1e-3)
        torch.nn.init.uniform_(self.l3.bias, -3*1e-3, 3*1e-3)

    def linear_batch_relu(self, i, o):
        return nn.Sequential(
            nn.Linear(i, o),
            nn.BatchNorm1d(o),
            nn.ReLU(),
        )

    def linear_relu(self, i, o):
        return nn.Sequential(
            nn.Linear(i, o),
            nn.ReLU(),
        )
        
    def forward(self, state, action):
        x = self.l1(state)
        x = torch.cat((x, action), dim=1)
        x = self.l2(x)
        x = self.l3(x)

        return x

class Critic_DQL(nn.Module):
    def __init__(self, batch:bool, nb_actions:int, state_space:int, seed:int=42) -> None:
        super(Critic_DQL, self).__init__()
        torch.manual_seed(seed)

        if batch:
            self.l1 = nn.Sequential(nn.BatchNorm1d(state_space), self.linear_batch_relu(state_space, 400))

        else:
            self.l1 = nn.Sequential(self.linear_relu(state_space, 400))

        self.l2 = self.linear_relu(400, 300)
        self.l3 = nn.Linear(300, nb_actions)

        torch.nn.init.uniform_(self.l3.weight, -3*1e-3, 3*1e-3)
        torch.nn.init.uniform_(self.l3.bias, -3*1e-3, 3*1e-3)

    def linear_batch_relu(self, i, o):
        return nn.Sequential(
            nn.Linear(i, o),
            nn.BatchNorm1d(o),
            nn.ReLU(),
        )

    def linear_relu(self, i, o):
        return nn.Sequential(
            nn.Linear(i, o),
            nn.ReLU(),
        )
        
    def forward(self, state):
        x = self.l1(state)
        x = self.l2(x)
        x = self.l3(x)

        return x

class Actor_DDPG(nn.Module):
    def __init__(self, batch:bool, state_space:int, seed:int=42) -> None:
        super(Actor_DDPG, self).__init__()
        torch.manual_seed(seed)

        if batch:
            self.l1 = nn.Sequential(nn.BatchNorm1d(state_space), self.linear_batch_relu(state_space, 400))
            self.l2 = self.linear_batch_relu(400, 300)

        else:
            self.l1 = nn.Sequential(self.linear_relu(state_space, 400))
            self.l2 = self.linear_relu(400, 300)

        last_layer = nn.Linear(300, 1)
        torch.nn.init.uniform_(last_layer.weight, -3*1e-3, 3*1e-3)
        torch.nn.init.uniform_(last_layer.bias, -3*1e-3, 3*1e-3)
        self.l3 = nn.Sequential(last_layer, nn.Tanh()) 

    def linear_batch_relu(self, i, o):
        return nn.Sequential(
            nn.Linear(i, o),
            nn.BatchNorm1d(o),
            nn.ReLU(),
        )

    def linear_relu(self, i, o):
        return nn.Sequential(
            nn.Linear(i, o),
            nn.ReLU(),
        )
        
    def forward(self, state):
        x = self.l1(state)
        x = self.l2(x)
        x = self.l3(x)
        return x