from python.constants import *
from python.domain import State, CarOnTheHillDomain
from python.policy import AlwaysAcceleratePolicy, RandomActionPolicy
from python.simulation import Simulation

import numpy as np


if __name__ == "__main__":
    domain = CarOnTheHillDomain(DISCOUNT_FACTOR, M, GRAVITY, TIME_STEP, INTEGRATION_TIME_STEP)
    policy = AlwaysAcceleratePolicy()
    #policy = RandomActionPolicy()
    initial_state = State.random_initial_state()

    simulation = Simulation(domain, policy, initial_state, remember_trajectory=True, seed=42)
    simulation.simulate(10)

    trajectory = np.array(simulation.get_trajectory())
    print(trajectory)
    
