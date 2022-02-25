from turtle import right
from python.Q_learner import Q_learn, Q_learn_estimation, Q_learn_temporal_difference
from python.components import Action
from python.components import State, StochasticDomain, LEFT, RIGHT, UP, DOWN

import numpy as np

from python.system_identification import MDP

class Policy():
    def chooseAction(self, state: State):
        pass 

    def J(self, domain: StochasticDomain, decay: float, N: int):
        """
        Computes the expected reward for every state 
        of the domain if we follow N steps of this policy.
        """
        m, n = domain.g.shape
        j_prec = np.zeros((n, m))

        for _ in range(N):
            j_curr = np.zeros((n, m))
            for x in range(n):
                for y in range(m):
                    state = State(x, y)
                    action = self.chooseAction(state)
                    for transition in domain.possibleTransitions(state, action):
                        new_state, reward, probability = transition
                        j_curr[x, y] += probability*(reward + decay*j_prec[new_state.x, new_state.y])

            j_prec = j_curr
        return j_curr

class AlwaysGoRightPolicy(Policy):
    def chooseAction(self, state: State):
        return RIGHT

class RandomUniformPolicy(Policy):
    def chooseAction(self, state: State):
        w = np.random.rand()
        if w <= 0.25:
            return RIGHT

        if w <= 0.5:
            return LEFT

        if w <= 0.75:
            return UP

        return DOWN

class QLearningPolicy(Policy):
    def __init__(self, domain: StochasticDomain, decay: float, N):
        self.Q = Q_learn(domain, decay, N)
        n, m = domain.g.shape
        self.Q_policy = np.array([domain.actions[np.argmax(self.Q[i, j, ])] for i in range(n) for j in range(m)]).reshape(n, m)

    def J(self, domain: StochasticDomain):
        """
        Computes the expected reward for every state 
        of the domain if we follow N steps of this policy.
        """
        n, m = domain.g.shape
        J = np.array([np.max(self.Q[i, j, ]) for i in range(n) for j in range(m)]).reshape(n, m)

        return J

    def chooseAction(self, state: State):
        return self.Q_policy[state.x, state.y]

class EstimatedQLearningPolicy(QLearningPolicy):
    def __init__(self, domain: StochasticDomain, mdp: MDP, decay: float, N):
        self.Q = Q_learn_estimation(domain, decay, N, mdp)
        n, m = domain.g.shape
        self.Q_policy = np.array([domain.actions[np.argmax(self.Q[i, j, ])] for i in range(n) for j in range(m)]).reshape(n, m)

class TrajectoryBasedQLearningPolicy(QLearningPolicy):
    def __init__(self, state_space, action_space, trajectory, learning_rate, decay, seed=42):
        self.Q = Q_learn_temporal_difference(state_space, action_space, trajectory, learning_rate, decay)
        n, m, _ = self.Q.shape 
        self.Q_policy = np.array([action_space[np.argmax(self.Q[i, j, ])] for i in range(n) for j in range(m)]).reshape(n, m)
