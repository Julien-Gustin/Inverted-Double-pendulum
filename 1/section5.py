
from python.components import DeterministicDomain, State, StochasticDomain
from python.constants import *
from python.policy import QLearningPolicy, RandomUniformPolicy, TrajectoryBasedQLearningPolicy
from python.simulation import Simulation

import math

from python.utils import get_max_diff_q, plot
from python.latex import *


if __name__ == "__main__":
    domains = [DeterministicDomain(G), StochasticDomain(G, W[0])]
    labels = ["Deterministic", "Stochastic"]
    initial_state = State(0, 3)
    policy = RandomUniformPolicy()

    latex = True

    epsilon = 1e-3
    N = math.ceil(math.log((epsilon / (2 * Br)) * (1. - GAMMA) ** 2, GAMMA))

    T = [10**i for i in range(1, 7)]
    learning_rate = 0.05
    
    for label, domain in zip(labels, domains):

        #QLearning using domain as being known     
        q_learning_policy = QLearningPolicy(domain, GAMMA, N)

        #generate a trajectory of size max(T)
        simulation = Simulation(domain, policy, initial_state, seed=256)
        trajectory = simulation.simulate(max(T))

        dist_between_qvalues = list()
        for t in T:
            #Trajectory based QLearning
            trajectory_based_policy = TrajectoryBasedQLearningPolicy(G, domain.actions, trajectory[:t], learning_rate, GAMMA, seed=42)
            dist_between_qvalues.append(get_max_diff_q(trajectory_based_policy.Q, q_learning_policy.Q))

        plot(T, dist_between_qvalues, "Q", "{}_trajectory_based_policy".format(label))

        print("Policy derived from trajectory based Q-learning: {}".format(label))
        print(trajectory_based_policy.Q_policy.T)

        print()

        print("Expected cumulative reward table: {}".format(label))
        print(trajectory_based_policy.J(domain))

        print("\n")

        if latex:
            print(matrix_to_table_string(trajectory_based_policy.Q_policy.T, "todo"))
            print(matrix_to_table(trajectory_based_policy.J(domain).T, "todo"))


    
