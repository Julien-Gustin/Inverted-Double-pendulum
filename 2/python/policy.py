import numpy as np
from python.domain import ACTIONS, State

class Policy():
    def make_action(self, state: State) -> int:
        pass 

class AlwaysAcceleratePolicy():
    def make_action(self, state: State) -> int:
        return 4

class RandomActionPolicy():
    def make_action(self, state: State) -> int:
        return np.random.choice(ACTIONS)