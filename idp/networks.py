import torch.nn as nn 
import torch 
import gym
import pybulletgym
from ddpg import DDPG 
import time 
import numpy as np
import random
# Seed
SEED = 3
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class Critic(nn.Module):
    def __init__(self) -> None:
        super(Critic, self).__init__()
        self.l1 = self.linear_batch_relu(9, 399)
        self.l2 = self.linear_batch_relu(399+1, 300)

        self.l3 = nn.Linear(300, 1)
        torch.nn.init.uniform_(self.l3.weight, -3*1e-3, 3*1e-3)

    def linear_batch_relu(self, i, o):
        return nn.Sequential(
            nn.Linear(i, o),
            nn.ReLU(),
            nn.BatchNorm1d(o),
        )
        
    def forward(self, state, action):
        x = self.l1(state)
        x = torch.cat((x, action), dim=1)
        x = self.l2(x)
        x = self.l3(x)
        return x
        

critic = Critic()

torch.manual_seed(SEED)

class Actor(nn.Module):
    def __init__(self) -> None:
        super(Actor, self).__init__()
        self.l1 = self.linear_batch_relu(9, 400)
        self.l2 = self.linear_batch_relu(400, 300)

        last_layer = nn.Linear(300, 1)
        torch.nn.init.uniform_(last_layer.weight, -3*1e-3, 3*1e-3)
        self.l3 = nn.Sequential(last_layer, nn.Tanh()) 
        

    def linear_batch_relu(self, i, o):
        return nn.Sequential(
            nn.Linear(i, o),
            nn.ReLU(),
            nn.BatchNorm1d(o),
        )
        
    def forward(self, state):
        x = self.l1(state)
        x = self.l2(x)
        x = self.l3(x)
        return x

actor = Actor()

gym.logger.set_level(40)
env = gym.make('InvertedDoublePendulumPyBulletEnv-v0')
env.seed(42)
# env.render() # call this before env.reset, if you want a window showing the environment
# env.reset()

ddpg = DDPG(env, critic, actor)
ddpg.apply()

torch.save(ddpg.target_actor, "actor")
torch.save(ddpg.target_critic, "critic")

state = env.reset()
while True:
    action = ddpg.target_actor(torch.as_tensor(state, dtype=torch.double))
    state, _, done, _ = env.step(action.detach().numpy())
    time.sleep(1e-2)

    if done:
       state = env.reset()