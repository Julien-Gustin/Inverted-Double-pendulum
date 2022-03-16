import numpy as np
from python.domain import ACTIONS, State
from python.fitted_Q import Fitted_Q
import torch 

class Policy():
    def make_action(self, state: State) -> int:
        pass 

class AlwaysAcceleratePolicy():
    def make_action(self, state: State) -> int:
        return 4

class AlwaysDesacceleratePolicy():
    def make_action(self, state: State) -> int:
        return -4

class RandomActionPolicy():
    def __init__(self, seed=42) -> None:
        np.random.seed(seed)

    def make_action(self, state: State) -> int:
        return np.random.choice(ACTIONS)

class FittedQPolicy():
    def __init__(self, fitted_q: Fitted_Q) -> None:
        self.fitted_q = fitted_q

    def make_action(self, state:State) -> int:
        return self.fitted_q.compute_optimal_actions(np.array([state.values()]))

class ParameterQPolicy():
    def __init__(self, parameter_Q):
        self.parameter_Q = parameter_Q

    def make_action(self, state:State):
        return self.parameter_Q.compute_optimal_actions(np.array([state.values()]))

class EpsilonGreedyPolicy():
    def __init__(self, parameter_Q, epsilon:float, seed):
        self.epsilon = epsilon
        self.parameter_Q = parameter_Q

        self.random_policy = RandomActionPolicy(seed)
        np.random.seed(seed)

    def make_action(self, state: State):
        noise = np.random.uniform()
        
        if noise <= self.epsilon:
            return self.random_policy.make_action(state)

        return self.parameter_Q.compute_optimal_actions(np.array([state.values()]))