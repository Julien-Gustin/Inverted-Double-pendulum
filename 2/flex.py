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
from section4 import get_trajectories, get_stopping_rules


    

if __name__ == "__main__":
    domain = CarOnTheHillDomain(DISCOUNT_FACTOR, M, GRAVITY, TIME_STEP, INTEGRATION_TIME_STEP)
    initial_state = State.random_initial_state()
    ETR = ExtraTreesRegressor(n_estimators=30, n_jobs=-1)

    epsilon = 1e-3
    N = math.ceil(math.log((epsilon / (2 * B_r)) * (1. - DISCOUNT_FACTOR), DISCOUNT_FACTOR))

    stopping_rule = get_stopping_rules()[0][0]
    trajectories = get_trajectories(1, 3)[1][0]

    fitted_Q = Fitted_Q(ETR, stopping_rule, DISCOUNT_FACTOR)
    fitted_Q.fit(trajectories)

    fitted_q_policy = FittedQPolicy(fitted_Q)

    simulation = Simulation(domain, fitted_q_policy, initial_state, remember_trajectory=True, seed=3)
    simulation.simulate(600)
    make_video("video/animation_fitted_Q_ETR.avi", simulation.get_trajectory())

