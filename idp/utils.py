import numpy as np
import argparse
import time

def parse_args():
    """ 
       Parse input arguments 
    """
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--ddpg', action="store_true") 
    parser.add_argument('--fqi', action="store_true")
    parser.add_argument('--dql', action="store_true")
    # Use batchnormalisation
    parser.add_argument('--batchnorm', action="store_true")
    # --render `model_file`
    parser.add_argument('--render', action="store")
    # discount factor
    parser.add_argument('--gamma', action='store', type=float)
    # Number of sample for fqi
    parser.add_argument('--samples', action='store', type=int)
    # Number of discrete action (should be odd)
    parser.add_argument('--actions', action='store', type=int)

    parser.add_argument('--seed', action='store', type=int)
    args = parser.parse_args()

    if args.actions is None:
        args.actions = 11

    if args.gamma is None:
        args.gamma = 0.99

    if args.seed is None:
        args.seed = 42

    if (args.ddpg + args.fqi + args.dql) != 1:
        print("Error: Should use either ddpg, fqi or dql using parameters\n\t`--ddpg` to use ddpg\n\t`--fqi` to use fitted Q iteration\n\t`--dql` to use deep-Q-learning")
        exit()

    if args.gamma < 0 or args.gamma > 1:
        print("Error: Please give a value for gamma between 0 and 1 with\n\t`--gamma` [GAMMA]")
        exit()

    if args.render is not None and (not args.ddpg and not args.dql): 
        print("Error: In order to render a given model you should use either `dql` or `ddpg`")
        exit()

    if not args.ddpg and (args.actions+1) % 2:
        print("Error: The number of discrete action should be odd")
        exit()

    if args.fqi and (args.samples is None):
        print("Error: Please provide a number of samples to compute FQI")
        exit()

    return args

def generate_sample(env, buffer_size:int, seed:int):
    """ Generate random trajectories """
    np.random.seed(seed)

    buffer = []
    prec_state = env.reset()

    while len(buffer) < buffer_size:
        action = env.action_space.sample() 
        state, reward, done, _ = env.step([action])

        buffer.append([prec_state, action, reward, state, done])
        if done:
            state = env.reset()

        prec_state = state
    return buffer[:buffer_size]

def get_discretize_action(n_actions:int):
    if n_actions % 2 == 0:
        print("error, should be even")
        exit()

    return np.linspace(-1, 1, n_actions)


def render(env, model):
    """ Render the double pendulum given a policy """
    env.render() 
    state = env.reset()

    while True:
        action = model.compute_optimal_actions(state)
        state, _, done, _ = env.step([action])
        time.sleep(1e-2)

        if done:
            state = env.reset()
