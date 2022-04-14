from python.constants import *
from python.domain import State, CarOnTheHillDomain
from python.policy import Policy
from python.simulation import Simulation

import numpy as np

def J_th(rewards, discount_factor):
    """ Take advantage of the problem statement to compute the expected return """
    t = np.where(rewards != 0)[0]
    if len(t) > 1:
        print("Error")

    if len(t) == 0:
        return 0

    t = t[0]

    return rewards[t] * discount_factor**t


def J(domain: CarOnTheHillDomain, policy: Policy, discount_factor:float, nb_simulations: int, trajectory_length:int, seed=0):
    """ Estimates the expected return of a policy for the car on the hill problem """
    J_hat = np.zeros((nb_simulations))

    for i in range(nb_simulations):
        initial_state = State.random_initial_state(seed=nb_simulations*seed + i)

        simulation = Simulation(domain, policy, initial_state, remember_trajectory=True, seed=nb_simulations*seed + i, stop_when_terminal=True)
        simulation.simulate(trajectory_length)

        trajectories = np.array(simulation.get_trajectory())

        rewards = trajectories[:, 2]
        J_hat[i] = J_th(rewards, discount_factor)

    return J_hat.mean()