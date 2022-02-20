from email import policy
import numpy as np

from python.constants import *
from python.components import State, DeterministicDomain, StochasticDomain
from python.policy import AlwaysGoRightPolicy
from python.simulation import Simulation
from python.latex import matrix_to_table

np.random.seed(42)

def simulateAndShowSingleTrajectory(initial_state: State, domain: StochasticDomain, steps: int):
    policy = AlwaysGoRightPolicy()
    simulation = Simulation(domain, policy, initial_state, 42)
    for i in range(steps):
        print("{}. ".format(i) ,simulation.step())

if __name__ == '__main__':
    initial_state = State(0, 3)
    print("\n--- Deterministic ---\n")
    domain = DeterministicDomain(G)

    simulateAndShowSingleTrajectory(initial_state, domain, 10)

    print("\n--- Stochastic ---\n")
    
    domain = StochasticDomain(G, W[0])
    simulateAndShowSingleTrajectory(initial_state, domain, 10)
