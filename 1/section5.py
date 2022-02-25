
from re import S
from python.components import DeterministicDomain, State, StochasticDomain
from python.constants import *
from python.policy import EpsilonGreedyPolicy, QLearningPolicy, RandomUniformPolicy, TrajectoryBasedQLearningPolicy
from python.simulation import Simulation

import math
import random

from python.utils import get_max_diff_q, plot
from python.latex import *

def online(episodes, transitions, learning_rate, epsilon, domains, labels, policies, initial_state, title, decay=1, replay=False):
    replay_buffer = []

    epsilon = 1e-3
    N = math.ceil(math.log((epsilon * (1.0-GAMMA))/ Br, GAMMA))

    for q_learning_policy, label, domain in zip(policies, labels, domains):

        policy = EpsilonGreedyPolicy(domain, learning_rate, epsilon)
        J_diff = []
        for episode in range(episodes):
            state = initial_state
            simulation = Simulation(domain, policy, state, seed=episode)

            for __ in range(transitions):
                trajectory_update = simulation.step()    

                if replay:
                    replay_buffer.append(trajectory_update)
                    for trajectory in random.choices(replay_buffer):
                        policy.updatePolicy(trajectory)
                else:
                    policy.updatePolicy(trajectory_update)

                policy.learning_rate *= decay

            J_diff.append(get_max_diff_q(policy.J(domain, GAMMA, N), q_learning_policy.J()))

        plot(range(episodes),J_diff, "J", "{}_{}".format(title, label), log=False, line_style="-", xlabel="Episode")


def _1(domains, labels, policies, initial_state):
    policy = RandomUniformPolicy()

    latex = False

    T = [10**i for i in range(1, 7)]
    learning_rate = 0.05
    
    for q_learning_policy, label, domain in zip(policies, labels, domains):

        #generate a trajectory of size max(T)
        simulation = Simulation(domain, policy, initial_state, seed=256)
        trajectory = simulation.simulate(max(T))

        dist_between_qvalues = list()
        for t in T:
            #Trajectory based QLearning
            trajectory_based_policy = TrajectoryBasedQLearningPolicy(domain, trajectory[:t], learning_rate, GAMMA, seed=42)
            dist_between_qvalues.append(get_max_diff_q(trajectory_based_policy.Q, q_learning_policy.Q))

        plot(T, dist_between_qvalues, "Q", "{}_trajectory_based_policy".format(label))

        print("Policy derived from trajectory based Q-learning: {}".format(label))
        print(trajectory_based_policy.Q_policy.T)

        print()

        print("Expected cumulative reward table: {}".format(label))
        print(trajectory_based_policy.J())

        print("\n")

        if latex:
            print(matrix_to_table_string(trajectory_based_policy.Q_policy.T, "todo"))
            print(matrix_to_table(trajectory_based_policy.J().T, "todo"))

def _2(domains, labels, policies, initial_state, title="5_2"):
    episodes = 100
    transitions = 1000
    learning_rate = 0.05
    epsilon = 0.5

    experiments = [
        {"title":"{}_experiment_1".format(title), "decay":1, "replay":False},
        {"title":"{}_experiment_2".format(title), "decay":0.8, "replay":False},
        {"title":"{}_experiment_3".format(title), "decay":1, "replay":True}
    ]

    for experiment in experiments:
        online(episodes, transitions, learning_rate, epsilon, domains, labels, policies, initial_state, **experiment)

def _3(domains, labels, initial_state):
    epsilon = 1e-3

    global GAMMA
    GAMMA = 0.4

    N = math.ceil(math.log((epsilon / (2 * Br)) * (1. - GAMMA) ** 2, GAMMA))

    for _, domain in zip(labels, domains):
        policies.append(QLearningPolicy(domain, GAMMA, N))

    _2(domains, labels, policies, initial_state, "5_3")



if __name__ == "__main__":
    domains = [DeterministicDomain(G), StochasticDomain(G, W[0])]
    labels = ["Deterministic", "Stochastic"]
    policies = []
    initial_state = State(0, 3)

    epsilon = 1e-3
    N = math.ceil(math.log((epsilon / (2 * Br)) * (1. - GAMMA) ** 2, GAMMA))

    for _, domain in zip(labels, domains):
        #QLearning using domain as being known     
        policies.append(QLearningPolicy(domain, GAMMA, N))


    _1(domains, labels, policies, initial_state)
    _2(domains, labels, policies, initial_state)
    _3(domains, labels, initial_state)
    
