import numpy as np

from python.constants import *
from python.components import State, DeterministicDomain, StochasticDomain
from python.policy import AlwaysGoRightPolicySimulation

def simulateAndShowSingleTrajectory(initial_state: State, domain: StochasticDomain, steps: int):
    simulation = AlwaysGoRightPolicySimulation(domain, initial_state.x, initial_state.y)
    for _ in range(steps):
        #simulation.domain.print_domain(simulation.state)
        print(simulation.step())

if __name__ == '__main__':
    initial_state = State(0, 3)
    domain = DeterministicDomain(G)

    simulateAndShowSingleTrajectory(initial_state, domain, 10)

    print("\n", "-"*50, "\n")

    domain = StochasticDomain(G, W[0])
    simulateAndShowSingleTrajectory(initial_state, domain, 10)
