import torch.nn as nn 
import torch 
import gym
import pybulletgym
from ddpg import DDPG 
import time 

critic = nn.Sequential(
    nn.Linear(10, 32, dtype=torch.double),
    nn.ReLU(),
    nn.Linear(32, 32, dtype=torch.double),
    nn.ReLU(),
    nn.Linear(32, 32, dtype=torch.double),
    nn.ReLU(),
    nn.Linear(32, 1, dtype=torch.double)
)

actor = nn.Sequential(
    nn.Linear(9, 32, dtype=torch.double),
    nn.ReLU(),
    nn.Linear(32, 32, dtype=torch.double),
    nn.ReLU(),
    nn.Linear(32, 32, dtype=torch.double),
    nn.ReLU(),
    nn.Linear(32, 1, dtype=torch.double)
)

gym.logger.set_level(40)
env = gym.make('InvertedDoublePendulumPyBulletEnv-v0')
env.render() # call this before env.reset, if you want a window showing the environment
env.reset()

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