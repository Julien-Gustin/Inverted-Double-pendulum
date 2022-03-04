from email import policy
from python.constants import *
from python.domain import State, CarOnTheHillDomain
from python.policy import AlwaysAcceleratePolicy, RandomActionPolicy
from python.simulation import Simulation

import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

#seed of the experiment
rs = RandomState(MT19937(SeedSequence(42)))

if __name__ == "__main__":
    domain = CarOnTheHillDomain(DISCOUNT_FACTOR, M, GRAVITY, TIME_STEP, INTEGRATION_TIME_STEP)
    policy = AlwaysAcceleratePolicy()
    #policy = RandomActionPolicy()
    initial_state = State.random_initial_state()

    simulation = Simulation(domain, policy, initial_state, remember_trajectory=True)
    simulation.simulate(10)

    trajectory = np.array(simulation.get_trajectory())
    print(trajectory)
    
