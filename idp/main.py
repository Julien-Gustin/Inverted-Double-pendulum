from ast import parse
import gym
from noise import OU  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import numpy as np
import time
import argparse

from utils import generate_sample
from fqi import Fitted_Q_ERT
from utils import *
from networks import Actor, Critic
from ddpg import DDPG
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ddpg', action="store_true")
    parser.add_argument('--fqi', action="store_true")
    parser.add_argument('--batch', action="store_true")
    parser.add_argument('--render', action="store")
    parser.add_argument('--gamma', action='store')
    parser.add_argument('--samples', action='store')
    args = parser.parse_args()

    if args.ddpg == args.fqi or args.gamma is None or (not args.ddpg and args.render is not None) or (args.fqi and not args.samples):
        print("Error: Should use either ddpg or fqi using parameters\n\t`--ddpg` to use ddpg\n\t`--fqi` to use fitted Q iteration")
        exit()

    return args


if __name__ == '__main__':
    # parse input
    args = parse_args()
    
    file_extension = "{}_{}_{}".format(args.batch, "ou", args.gamma)

    # launch environment
    gym.logger.set_level(40)
    env = gym.make('InvertedDoublePendulumPyBulletEnv-v0')
    env.seed(42)

    if args.fqi:
        actions = get_discretize_action(11)
        samples = generate_sample(env, int(args.samples), actions, seed=args.samples)
        fqi = Fitted_Q_ERT(float(args.gamma), actions, seed=args.samples)
        fqi.fit(samples)

        mean, std = J(env, fqi, 0.99, 50, 1000)
        print("{} samples: | mean: {} | std: {}".format(args.samples, mean, std))

    if args.ddpg:
        actor = Actor(batch=bool(args.batch), state_space=9)
        critic = Critic(batch=bool(args.batch), action_space=1, state_space=9)

        # Render a loaded model
        if args.render is not None:
            env.render() 
            state = env.reset()
            actor.load_state_dict(torch.load(args.render, map_location="cpu"))
            ddpg = DDPG(env, critic, actor, OU(0, 0, 0.15, 0.2), file_extension ,gamma=float(args.gamma))

            while True:
                action = ddpg.compute_optimal_actions([state])
                state, _, done, _ = env.step([action])
                time.sleep(1e-2)

                if done:
                    state = env.reset()

        # Train ddpg
        else:
            ddpg = DDPG(env, critic, actor, OU(0, 0, 0.15, 0.2), file_extension ,gamma=float(args.gamma))
            ddpg.apply()

