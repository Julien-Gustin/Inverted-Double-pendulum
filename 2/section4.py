from python.constants import *
from python.domain import ACTIONS, State, CarOnTheHillDomain
from python.policy import AlwaysAcceleratePolicy, RandomActionPolicy, FittedQPolicy
from python.simulation import Simulation
from python.fitted_Q import Fitted_Q

import numpy as np

from python.constants import *
from python.domain import CarOnTheHillDomain
from python.expected_return import J

import math
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from section3 import make_video

def get_stopping_rules():
    epsilon = 1e-3
    N = math.ceil(math.log((epsilon / (2 * B_r)) * (1. - DISCOUNT_FACTOR), DISCOUNT_FACTOR))

    return N, None

def get_models():
    LR = LinearRegression()
    ETR = ExtraTreesRegressor(n_estimators=50, random_state=42)
    # TODO: MLP

    return LR, ETR

def get_trajectories(trajectory_length, N):
    # Random
    random_policy = RandomActionPolicy()
    random_trajectories = np.array([])

    random_trajectories = []

    for n in range(N):
        initial_state = State.random_initial_state(seed=n)
        simulation = Simulation(domain, random_policy, initial_state, remember_trajectory=True, seed=n)
        simulation.simulate(trajectory_length)
        random_trajectories.append(np.array(simulation.get_trajectory(values=True)).squeeze())

    random_trajectories = np.concatenate(np.array(random_trajectories), axis=1)


    # TODO:

    return random_trajectories, None

def plot_Q():
    pass

def plot_mu():
    pass

def plot_J_mu():
    pass
    

if __name__ == "__main__":
    domain = CarOnTheHillDomain(DISCOUNT_FACTOR, M, GRAVITY, TIME_STEP, INTEGRATION_TIME_STEP)

    # grid = 
    p = np.linspace(-1, 1, 100)
    s = np.linspace(-3, 3, 300)

    pp, ss = np.meshgrid(p[:-1], s[:-1], indexing='ij')
    states = np.vstack((pp.ravel(), ss.ravel())).T

    gridshape = pp.shape

    for trajectories in get_trajectories(1000, 100):
        for stopping_rule in get_stopping_rules():
            for model in get_models():
                fitted_Q = Fitted_Q(model, stopping_rule, DISCOUNT_FACTOR)
                fitted_Q.fit(trajectories)

                y_pred = fitted_Q.compute_optimal_actions(states).reshape(gridshape)

                plt.pcolormesh(
                    p[:-1], s[:-1], y_pred.T,
                    cmap='coolwarm_r',
                    vmin=-4, vmax=4,
                    rasterized=True,
                    shading="auto",
                )
                plt.xlabel(r'$p$')
                plt.ylabel(r'$s$')

                plt.savefig(str(model.__class__) + ".png")
                plt.close()




