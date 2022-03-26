import gym
from ddpg import DDPG
from networks import critic, actor

env = gym.make('InvertedDoublePendulumPyBulletEnv-v0')
ddpg = DDPG(env, critic, actor)
ddpg.apply()