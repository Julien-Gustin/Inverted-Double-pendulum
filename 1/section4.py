import matplotlib.pyplot as plt
import math
import numpy as np
from python.latex import matrix_to_table, matrix_to_table_string

from python.system_identification import MDP
from python.policy import EstimatedQLearningPolicy, QLearningPolicy, RandomUniformPolicy
from python.components import DeterministicDomain, StochasticDomain, State
from python.simulation import Simulation
from python.constants import *
from python.utils import *


if __name__ == '__main__':
    np.random.seed(12345)
    epsilon = 1e-3
    N = math.ceil(math.log((epsilon / (2 * Br)) * (1. - GAMMA) ** 2, GAMMA))
    initial_state = State(0, 3)
    show_latex = True

    deterministic_domain = DeterministicDomain(G)
    stochastic_domain = StochasticDomain(G, W[0])

    n, m = deterministic_domain.g.shape

    policy = RandomUniformPolicy()

    Q_policy_deterministic = QLearningPolicy(deterministic_domain, 0.99, N)
    Q_policy_stochastic = QLearningPolicy(stochastic_domain, 0.99, N)

    T = [1, 10, 10**2, 10**3, 10**4, 10**5, 10**6, 10**7]

    print("N =", N)

    for domain_name, domain, Q_policy in [("Deterministic", deterministic_domain, Q_policy_deterministic),
                                          ("Stochastic", stochastic_domain, Q_policy_stochastic)]:
        print("\n--- {} ---\n".format(domain_name))
        print("Value function:\n",Q_policy.J().T)
        if show_latex:
            print(matrix_to_table(Q_policy.J().T, "TODO"))

        R_diff = []
        P_diff = []
        Q_diff = []

        simulation = Simulation(domain, policy, initial_state, 12345)

        h = simulation.simulate(T[-1])

        for t in T:
            mdp = MDP(n, m, len(domain.actions), h[:t])

            estimated_Q_policy = EstimatedQLearningPolicy(domain, mdp, 0.99, N)

            R_diff.append(get_max_diff_r(domain, mdp))
            P_diff.append(get_max_diff_p(domain, mdp))
            Q_diff.append(infinity_norm(estimated_Q_policy.Q, Q_policy.Q))

            print("\nJ^N estimation after {} step:\n".format(t), estimated_Q_policy.J(domain, 0.99, N).T)

        if show_latex:
            print(matrix_to_table_string(estimated_Q_policy.Q_policy.T, "TODO"))
            print(matrix_to_table(estimated_Q_policy.J(domain, 0.99, N).T, "TODO"))

        plot(T, R_diff, "r", "{}_R".format(domain_name))
        plot(T, P_diff, "p", "{}_P".format(domain_name))
        plot(T, Q_diff, "Q", "{}_Q".format(domain_name))
