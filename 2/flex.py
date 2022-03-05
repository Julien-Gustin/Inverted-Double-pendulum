from python.constants import *
from python.domain import State, CarOnTheHillDomain
from python.policy import AlwaysAcceleratePolicy, RandomActionPolicy, FittedQPolicy
from python.simulation import Simulation
from python.fitted_Q import Fitted_Q

import numpy as np

from python.constants import *
from python.domain import CarOnTheHillDomain

import math

from sklearn.ensemble import ExtraTreesRegressor
from section3 import make_video

def get_stopping_rule():
    epsilon = 1e-3
    N = math.ceil(math.log((epsilon / (2 * B_r)) * (1. - DISCOUNT_FACTOR), DISCOUNT_FACTOR))

    return N



def get_trajectories(trajectory_length, N):
    # Random
    random_policy = RandomActionPolicy()
    random_trajectories = np.array([])

    random_trajectories = []

    for n in range(N):
        initial_state = State.random_initial_state(seed=n)
        simulation = Simulation(domain, random_policy, initial_state, remember_trajectory=True, seed=n, stop_when_terminal=True)
        simulation.simulate(trajectory_length)
        random_trajectories.extend(np.array(simulation.get_trajectory(values=True)).squeeze())
        print(n, "/", N)

    random_trajectories = np.array(random_trajectories)
    return random_trajectories
    

if __name__ == "__main__":
    domain = CarOnTheHillDomain(DISCOUNT_FACTOR, M, GRAVITY, TIME_STEP, INTEGRATION_TIME_STEP)
    initial_state = State.random_initial_state()
    ETR = ExtraTreesRegressor(n_estimators=30, n_jobs=-1)
    N = get_stopping_rule()
    trajectories = get_trajectories(500, 500)

    fitted_Q = Fitted_Q(ETR, N, DISCOUNT_FACTOR)
    fitted_Q.fit(trajectories)

    fitted_q_policy = FittedQPolicy(fitted_Q)

    simulation = Simulation(domain, fitted_q_policy, initial_state, remember_trajectory=True, seed=3)
    simulation.simulate(600)
    make_video("video/animation_fitted_Q_ETR.avi", simulation.get_trajectory())

