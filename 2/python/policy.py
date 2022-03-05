import numpy as np
from python.domain import ACTIONS, State
from python.fitted_Q import Fitted_Q

class Policy():
    def make_action(self, state: State) -> int:
        pass 

class AlwaysAcceleratePolicy():
    def make_action(self, state: State) -> int:
        return 4

class RandomActionPolicy():
    def make_action(self, state: State) -> int:
        return np.random.choice(ACTIONS)

class FittedQPolicy():
    def __init__(self, fitted_q: Fitted_Q) -> None:
        self.fitted_q = fitted_q

    def make_action(self, state:State) -> int:
        return self.fitted_q.compute_optimal_actions(np.array([state.values()]))