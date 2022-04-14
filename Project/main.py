import gym
import pybulletgym  # register PyBullet enviroments with open ai gym

import torch

from models.dql import DQL
from models.utils.noise import OU 
from models.utils.expected_return import J
from utils import generate_sample, parse_args
from models.fqi import Fitted_Q_ERT
from utils import *
from models.networks import Actor_DDPG, Critic_DDPG, Critic_DQL
from models.ddpg import DDPG

def launch_FQI(env, args, actions):
    """ Fitted Q-iteration algorithm """
    samples = generate_sample(env, args.samples, seed=args.seed)
    fqi = Fitted_Q_ERT(args.gamma, actions, env, seed=args.seed)
    fqi.fit(samples, compute_j=True)
    fqi.make_plot()

def launch_DDPG(env, args, file_extension): 
    """ Deep deterministic policy gradient """
    actor = Actor_DDPG(batch=args.batchnorm, state_space=9, seed=args.seed)
    critic = Critic_DDPG(batch=args.batchnorm, action_space=1, state_space=9, seed=args.seed)

    # Render a loaded model
    if args.render is not None:
        actor.load_state_dict(torch.load(args.render, map_location="cpu"))
        ddpg = DDPG(env, critic, actor, OU(0, 0, 0.15, 0.2), file_extension ,gamma=args.gamma)

        render(env, ddpg)

    # Train ddpg
    else:
        ddpg = DDPG(env, critic, actor, OU(0, 0, 0.15, 0.2), file_extension ,gamma=args.gamma)
        ddpg.apply()

def launch_DQL(env, args, actions, file_extension):
    """ Deep Q-Learning """
    critic = Critic_DQL(args.batchnorm, len(actions), state_space=9, seed=args.seed)

    # Render a loaded model    
    if args.render is not None:
        critic.load_state_dict(torch.load(args.render, map_location="cpu"))
        dql = DQL(env, critic, file_extension, actions=actions, gamma=args.gamma)

        render(env, dql)

    # Train dql
    else:
        dql = DQL(env, critic, file_extension, actions=actions, gamma=args.gamma)
        dql.apply()

if __name__ == '__main__':
    # parse input
    args = parse_args()
    if args.actions is not None:
        n_actions = args.actions
        actions = get_discretize_action(n_actions)
        file_extension = "{}_{}_{}".format(args.batchnorm, args.actions, args.gamma)

    else:
        file_extension = "{}_{}_{}".format(args.batchnorm, args.gamma)

    # launch environment
    gym.logger.set_level(40)
    env = gym.make('InvertedDoublePendulumPyBulletEnv-v0')
    env.seed(42)

    # Fitted Q-Iteration
    if args.fqi:
        launch_FQI(env, args, actions)

    # Deep Deterministic Policy Gradient
    if args.ddpg:
        launch_DDPG(env, args, file_extension)

    # Deep Q-Learning
    if args.dql:
        launch_DQL(env, args, actions, file_extension)

