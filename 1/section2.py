from python.policy import AlwaysGoRightPolicy
from python.constants import *
from python.components import StochasticDomain, DeterministicDomain
from python.latex import matrix_to_table

import numpy as np
import math

np.random.seed(42)

if __name__ == '__main__':
    epsilon = 1e-6
    N = math.ceil(math.log((epsilon * (1.0-GAMMA))/ Br, GAMMA))
    show_latex = False
    print("N =", N)

    policy = AlwaysGoRightPolicy()

    print("\n--- Deterministic ---\n")
    domain = DeterministicDomain(G)
    print(policy.J(domain, GAMMA, N).T)

    if show_latex:
        print(matrix_to_table(policy.J(domain, GAMMA, N).T, "$J^{\\mu}_N(x, y)$ for all $(x, y) \\in X$ in the deterministic domain; $N = " + str(N) + "$"))

    print("\n--- Stochastic ---\n")
    domain = StochasticDomain(G, W[0])
    print(policy.J(domain, GAMMA, N).T)

    if show_latex:
        print(matrix_to_table(policy.J(domain, GAMMA, N).T, "$J^{\\mu}_N(x, y)$ for all $(x, y) \\in X$ in the stochastic domain; $N = " + str(N) + "$"))