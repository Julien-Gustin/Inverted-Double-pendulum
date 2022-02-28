
from re import S
from python.components import DeterministicDomain, State, StochasticDomain
from python.constants import *
from python.policy import EpsilonGreedyPolicy, QLearningPolicy, RandomUniformPolicy, TrajectoryBasedQLearningPolicy
from python.simulation import Simulation

import math
import random
import matplotlib.pyplot as plt

from python.utils import infinity_norm, plot_with_std
from python.latex import *

import matplotlib.pylab as plt
import numpy as np

def online(episodes, transitions, learning_rate, e, domains, labels, policies, initial_state, gamma, title, decay=1, replay=False):
    epsilon = 1e-3
    N = math.ceil(math.log((epsilon / (2 * Br)) * (1. - gamma) ** 2, gamma))

    for q_learning_policy, label, domain in zip(policies, labels, domains):
        nb_iterations = 10


        dist_between_jvalues = np.zeros((nb_iterations, episodes))
        dist_between_q = np.zeros((nb_iterations, episodes))
        matrix_reach = np.zeros(domain.g.shape)

        # Iterate to compute mean and std, incorpore uncertanty
        for i in range(nb_iterations):
            replay_buffer = []
            policy = EpsilonGreedyPolicy(domain, learning_rate, e, seed=i)

            for episode in range(episodes): 
                state = initial_state
                simulation = Simulation(domain, policy, state, seed=nb_iterations*i+episode)

                for __ in range(transitions):
                    trajectory_update = simulation.step()    
                    matrix_reach[trajectory_update[0].x, trajectory_update[0].y] += 1

                    if replay: 
                        replay_buffer.append(trajectory_update)
                        for trajectory in random.choices(replay_buffer, k=10):
                            policy.updatePolicy(trajectory, gamma)

                    else:
                        policy.updatePolicy(trajectory_update, gamma)

                    policy.learning_rate *= decay

                policy.epsilon = 0
                dist_between_jvalues[i][episode] = infinity_norm(policy.J(domain, gamma, N), q_learning_policy.J())
                dist_between_q[i][episode] = infinity_norm(policy.Q, q_learning_policy.Q)
                policy.epsilon = e

                policy.learning_rate = learning_rate

        means_j = dist_between_jvalues.mean(axis=0)
        stds_j = dist_between_jvalues.std(axis=0)

        means_q = dist_between_q.mean(axis=0)
        stds_q = dist_between_q.std(axis=0)

        plot_with_std(range(episodes), means_j, stds_j, title + "_" + label, r'$\left\| J^N_{\mu_\hat{Q}} - J^N_{\mu^*} \right\|_\infty$', log=False, line_style="-", xlabel="Episode")
        plot_with_std(range(episodes), means_q, stds_q, title + "_" + label + "_Q", r'$\left\| \hat{Q} - Q^N \right\|_\infty$', log=False, line_style="-", xlabel="Episode")

def _1(domains, labels, policies, initial_state):
    latex = False

    T = [10**i for i in range(1, 7)]
    learning_rate = 0.05

    epsilon = 1e-3
    N = math.ceil(math.log((epsilon / (2 * Br)) * (1. - GAMMA) ** 2, GAMMA))
    
    for q_learning_policy, label, domain in zip(policies, labels, domains):
        nb_iterations = 10

        n, m = domain.g.shape

        J = np.zeros((n, m, len(T), nb_iterations))
        Q = np.zeros((n, m, len(domain.actions), len(T), nb_iterations))
        
        dist_between_qvalues = np.zeros((nb_iterations, len(T)))

        for i in range(nb_iterations): 
            policy = RandomUniformPolicy(seed=i*10)
            
            #generate a trajectory of size max(T)
            simulation = Simulation(domain, policy, initial_state, seed=i)
            trajectory = simulation.simulate(max(T))
            
            for ti,t in enumerate(T):
                #Trajectory based QLearning
                trajectory_based_policy = TrajectoryBasedQLearningPolicy(domain, trajectory[:t], learning_rate, GAMMA)
                dist_between_qvalues[i][ti] = infinity_norm(trajectory_based_policy.Q, q_learning_policy.Q)
                Q[:,:,:, ti, i] = trajectory_based_policy.Q
                trajectory_based_policy.compute_policy()
                J[:,:, ti, i] = trajectory_based_policy.J(domain, GAMMA, N)

        Q_means = Q[:,:,:, len(T)-1, :].mean(axis=-1)

        J_means = J[:,:, len(T)-1, :].mean(axis=-1)
        J_stds = J[:,:, len(T)-1, :].std(axis=-1)

        trajectory_based_policy.Q = Q_means
        trajectory_based_policy.compute_policy()

        means = dist_between_qvalues.mean(axis=0)
        stds = dist_between_qvalues.std(axis=0)

        plot_with_std(T, means, stds, "5_1_{}".format(label), r'$\left\| \hat{Q} - {Q}_N \right\|_\infty$')

        print("\n--- {} ---\n".format(label))

        print("Policy derived from trajectory based Q-learning: {}".format(label))
        print(trajectory_based_policy.Q_policy.T)

        print()

        print("Expected cumulative reward table: {}".format(label))
        print(J_means.T)

        print("\n")

        if latex:
            print(matrix_to_table_string(trajectory_based_policy.Q_policy.T, "todo"))
            print(matrix_to_table(J_means.T, "mean"))
            print(matrix_to_table(J_stds.T, "std"))

def _2(domains, labels, policies, initial_state, title="5_2", gamma=GAMMA):
    episodes = 100
    transitions = 1000
    learning_rate = 0.05
    e = 0.5

    experiments = [
        {"title":"{}_experiment_1".format(title), "decay":1, "replay":False},
        {"title":"{}_experiment_2".format(title), "decay":0.8, "replay":False},
        {"title":"{}_experiment_3".format(title), "decay":1, "replay":True}
    ]

    for experiment in experiments:
        print(" {}".format(experiment["title"]))
        online(episodes, transitions, learning_rate, e, domains, labels, policies, initial_state, gamma=gamma, **experiment)

def _3(domains, labels, initial_state):
    epsilon = 1e-3
    episodes = 100
    transitions = 1000
    learning_rate = 0.05
    e = 0.5

    policies = []

    gamma = 0.4

    N = math.ceil(math.log((epsilon / (2 * Br)) * (1. - gamma) ** 2, gamma))

    for _, domain in zip(labels, domains):
        policies.append(QLearningPolicy(domain, gamma, N))

    experiment = {"title":"5_3_experiment_1", "decay":1, "replay":False}
    online(episodes, transitions, learning_rate, e, domains, labels, policies, initial_state, gamma=gamma, **experiment)




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


    # _1(domains, labels, policies, initial_state)
    _2(domains, labels, policies, initial_state)
    _3(domains, labels, initial_state)
    
