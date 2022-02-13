from pickle import TRUE
from python.policy import AlwaysGoRightPolicySimulation, PolicySimulation
from python.constants import *
from python.components import StochasticDomain, DeterministicDomain, State

import numpy as np
import math
import copy

if __name__ == '__main__':
    epsilon = 1e-6
    N = math.ceil(math.log((epsilon * (1.0 -GAMMA))/ Br, GAMMA))
    print("N =", N)

    print("\n--- Deterministic ---\n")
    domain = DeterministicDomain(G)
    simulation = AlwaysGoRightPolicySimulation(domain, 0, 0)
    print(simulation.J(N))

    print("\n--- Stochastic ---\n")
    domain = StochasticDomain(G, W[0])
    simulation = AlwaysGoRightPolicySimulation(domain, 0, 0)
    print(simulation.J(N))
