from ast import parse
import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import numpy as np
import time
import argparse

from utils import generate_sample
from fqi import Fitted_Q_ERT
from utils import *
from networks import Actor, Critic
from ddpg import DDPG

# actions = get_discretize_action(11)

# fqi = None
# for size in [10, 100, 1000, 5000, 10000, 50000, 100000, 200000, 500000]:
#     samples = generate_sample(env, size, actions, seed=size)
#     fqi = Fitted_Q_ERT(0.95, actions, seed=size)
#     fqi.fit(samples)

#     m, s = J(env, fqi, 0.99, 50, 1000)
#     print("{} samples: | mean: {} | std: {}".format(size, m, s))

# # samples = generate_sample(env, 500000, actions, seed=42)
# # fqi = Fitted_Q_ERT(0.95, actions)
# # fqi.fit(samples)
# # print("J:", J(env, fqi, 0.95, 50, 1000))
# state = env.reset()
# while True:
#     action = fqi.compute_optimal_actions([state])
#     state, _, done, _ = env.step([action])
#     time.sleep(1e-2)

#     if done:
#        state = env.reset()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ddpg', action="store_true")
    parser.add_argument('--fqi', action="store_true")
    parser.add_argument('--batch', action="store_true")
    parser.add_argument('--render', action="store_true")
    args = parser.parse_args()

    if args.ddpg == args.fqi:
        print("Error: Should use either ddpg or fqi using parameters\n\t`--ddpg` to use ddpg\n\t`--fqi` to use fitted Q iteration")
        exit()

    return args


if __name__ == '__main__':

    # parse input
    args = parse_args()

    print(args.ddpg)


    gym.logger.set_level(40)
    env = gym.make('InvertedDoublePendulumPyBulletEnv-v0')

    # if render

    # env.render() # call this before env.reset, if you want a window showing the environment
    # env.reset()

    if args.fqi:
        pass

    # fqi

    # ddpg

    if args.ddpg:
        pass

    # ddpg = True

    # if ddpg:
    #     actor = Actor(batch=True, state_space=9)
    #     critic = Critic(batch=True, action_space=1, state_space=9)

    #     ddpg = DDPG(env, critic, actor)
    #     ddpg.apply()