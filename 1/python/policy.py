from python.Q_learner import Q_learn, Q_learn_estimation
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

class Simulation():
    def __init__(self, domain: StochasticDomain, policy: Policy, state: State, seed:int):
        self.domain = domain
        self.state = state
        self.policy = policy 
        np.random.seed(seed)

    def step(self):
        action = self.policy.chooseAction(self.state)
        prev_state = self.state
        self.state, reward = self.domain.interact(self.state, action)
        return prev_state, action, reward, self.state

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
        self.Q_policy = np.array([domain.actions[np.argmax(self.Q[i,j, ])] for i in range(n) for j in range(m)]).reshape(n, m)

    def J(self, domain: StochasticDomain):
        """
        Computes the expected reward for every state 
        of the domain if we follow N steps of this policy.
        """
        n, m = domain.g.shape
        J = np.array([np.max(self.Q[i,j, ]) for i in range(n) for j in range(m)]).reshape(n, m)

        return J

    def chooseAction(self, state: State):
        return self.Q_policy[state.x, state.y]

class EstimatedQLearningPolicy(QLearningPolicy):
    def __init__(self, domain: StochasticDomain, mdp: MDP, decay: float, N):
        self.Q = Q_learn_estimation(domain, decay, N, mdp)
        n, m = domain.g.shape
        self.Q_policy = np.array([domain.actions[np.argmax(self.Q[i, j, ])] for i in range(n) for j in range(m)]).reshape(n, m)

class TrajectoryBasedQLearningPolicy(QLearningPolicy):
    def _oneStepUpdate(self, transition, action_space, learning_ratio, decay):
        starting_state, action, reward, new_state = transition
        action_index = action_space.index(action)
        self.Q[starting_state.x, starting_state.y, action_index] += learning_ratio*(reward + decay*max(self.Q[new_state.x, new_state.y, ])
        - self.Q[starting_state.x, starting_state.y, action_index])

    def __init__(self, domain: StochasticDomain, initial_state: State, trajectory_steps, learning_ratio, decay, seed=42):
        n, m = domain.g.shape
        nb_actions = len(domain.actions)
        self.Q = np.zeros((n, m, nb_actions), dtype=float)

        #initialize trajectory simulator
        trajectory_simulation = Simulation(domain, RandomUniformPolicy(), initial_state, seed=seed)

        for _ in range(trajectory_steps):
            prev_state, action, reward, new_state = trajectory_simulation.step()
            transition = (prev_state, action, reward, new_state)
            self._oneStepUpdate(transition, domain.actions, learning_ratio, decay)

        self.Q_policy = np.array([domain.actions[np.argmax(self.Q[i, j, ])] for i in range(n) for j in range(m)]).reshape(n, m)
