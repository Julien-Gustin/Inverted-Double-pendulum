import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import numpy as np
import time

from utils import generate_sample
from fqi import Fitted_Q_ERT
from utils import *

gym.logger.set_level(40)
env = gym.make('InvertedDoublePendulumPyBulletEnv-v0')
env.render() # call this before env.reset, if you want a window showing the environment
env.reset()

actions = get_discretize_action(11)

fqi = None
for size in [10, 100, 1000, 5000, 10000, 50000, 100000, 200000, 500000]:
    samples = generate_sample(env, size, actions, seed=size)
    fqi = Fitted_Q_ERT(0.95, actions, seed=size)
    fqi.fit(samples)

    m, s = J(env, fqi, 0.99, 50, 1000)
    print("{} samples: | mean: {} | std: {}".format(size, m, s))

# samples = generate_sample(env, 500000, actions, seed=42)
# fqi = Fitted_Q_ERT(0.95, actions)
# fqi.fit(samples)
# print("J:", J(env, fqi, 0.95, 50, 1000))
state = env.reset()
while True:
    action = fqi.compute_optimal_actions([state])
    state, _, done, _ = env.step([action])
    time.sleep(1e-2)

    if done:
       state = env.reset()