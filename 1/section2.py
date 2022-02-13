from email import policy
from python.policy import AlwaysGoRightPolicy
from python.constants import *
from python.components import StochasticDomain, DeterministicDomain

import math

if __name__ == '__main__':
    epsilon = 1e-6
    N = math.ceil(math.log((epsilon * (1.0 -GAMMA))/ Br, GAMMA))
    print("N =", N)

    policy = AlwaysGoRightPolicy()

    print("\n--- Deterministic ---\n")
    domain = DeterministicDomain(G)
    print(policy.J(domain, GAMMA, N))

    print("\n--- Stochastic ---\n")
    domain = StochasticDomain(G, W[0])
    print(policy.J(domain, GAMMA, N))
