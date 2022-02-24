import matplotlib.pyplot as plt
import math
import numpy as np
from python.latex import matrix_to_table, matrix_to_table_string

from python.system_identification import MDP
from python.policy import EstimatedQLearningPolicy, QLearningPolicy, RandomUniformPolicy, Simulation
from python.components import DeterministicDomain, StochasticDomain, State
from python.constants import *

plt.rcParams['font.size'] = 14

flags = {
    'bbox_inches': 'tight'
}

def get_max_diff_r(domain, mdp):
    """Infinity norm of r"""
    n, m = domain.g.shape
    max_val = -1
    for x in range(n):
        for y in range(m):
            for action in domain.actions:
                diff = abs(mdp.r(State(x, y), action) - domain.r(State(x, y), action))
                if diff > max_val:
                    max_val = diff

    return max_val

def get_max_diff_p(domain, mdp):
    """Infinity norm of p"""
    n, m = domain.g.shape
    max_val = -1
    for x in range(n):
        for y in range(m):
            for action in domain.actions:
                for x_given in range(n):
                    for y_given in range(m):
                        diff = abs(mdp.p(State(x, y), State(x_given, y_given), action) - domain.p(State(x, y), State(x_given, y_given), action))
                        if diff > max_val:
                            max_val = diff

    return max_val

def get_max_diff_q(Q_hat, Q):
    """Infinity norm of q"""
    max_val = np.max(np.abs(Q_hat - Q))

    return max_val

def plot(x, y, estimate:str, file_name:str):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(x, y, '--bo')
    plt.xscale('log')
    plt.xlabel('$t$')
    plt.ylabel(f'$\\left\\| \\hat{{}} - {{}} \\right\\|_\\infty$'.format(estimate, estimate))
    plt.grid()
    plt.savefig("figures/{}".format(file_name), **flags)

    plt.close()



if __name__ == '__main__':
    np.random.seed(42)
    epsilon = 1e-3
    N = math.ceil(math.log((epsilon / (2 * Br)) * (1. - GAMMA) ** 2, GAMMA))
    initial_state = State(0, 3)
    show_latex = False

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
        print("Value function:\n",Q_policy.J(domain).T)
        if show_latex:
            print(matrix_to_table(Q_policy.J(domain).T, "TODO"))

        R_diff = []
        P_diff = []
        Q_diff = []

        for t in T:
            mdp = MDP(n, m, len(domain.actions))
            simulation = Simulation(domain, policy, initial_state, 42)
            for _ in range(t):
                state, action, reward, next_state = simulation.step()
                mdp.update(state, action, reward, next_state)

            estimated_Q_policy = EstimatedQLearningPolicy(domain, mdp, 0.99, N)

            R_diff.append(get_max_diff_r(domain, mdp))
            P_diff.append(get_max_diff_p(domain, mdp))
            Q_diff.append(get_max_diff_q(estimated_Q_policy.Q, Q_policy.Q))

            print("\nValue function estimation after {} step:\n".format(t), estimated_Q_policy.J(domain).T)

        if show_latex:
            print(matrix_to_table_string(estimated_Q_policy.Q_policy.T, "TODO"))
            print(matrix_to_table(estimated_Q_policy.J(domain).T, "TODO"))

        plot(T, R_diff, "r", "{}_R".format(domain_name))
        plot(T, P_diff, "p", "{}_P".format(domain_name))
        plot(T, Q_diff, "Q", "{}_Q".format(domain_name))
