import math 
import numpy as np

from python.constants import *
from python.components import DeterministicDomain, StochasticDomain
from python.policy import QLearningPolicy
from python.latex import matrix_to_table, matrix_to_table_string

np.random.seed(42)

if __name__ == '__main__':
    epsilon = 1e-3
    N = math.ceil(math.log((epsilon / (2 * Br)) * (1. - GAMMA) ** 2, GAMMA))
    show_latex = False

    print("N =", N)

    print("\n--- Deterministic ---\n")

    deterministic_domain = DeterministicDomain(G)
    qlearning_policy = QLearningPolicy(deterministic_domain, GAMMA, N)
    print("Value function:\n",qlearning_policy.J().T)
    print("\nPolicy:\n",qlearning_policy.Q_policy.T)

    if show_latex:
        print(matrix_to_table_string(qlearning_policy.Q_policy.T, "$\\mu^*_N$ in the deterministic domain; $N = " + str(N) + "$"))
        print(matrix_to_table(qlearning_policy.J().T, "$J_{\\mu^*}^N(x, y)$ for all $(x, y) \\in X$ in the deterministic domain; $N = " + str(N) + "$"))


    print("\n--- Stochastic ---\n")

    stochastic_domain = StochasticDomain(G, W[0])
    qlearning_policy = QLearningPolicy(stochastic_domain, GAMMA, N)
    print("Value function:\n", qlearning_policy.J().T)
    print("\nPolicy:\n",qlearning_policy.Q_policy.T)

    if show_latex:
        print(matrix_to_table_string(qlearning_policy.Q_policy.T, "$\\mu^*_N$ in the stochastic domain; $N = " + str(N) + "$"))
        print(matrix_to_table(qlearning_policy.J().T, "$J_{\\mu^*}^N(x, y)$ for all $(x, y) \\in X$ in the stochastic domain; $N = " + str(N) + "$"))