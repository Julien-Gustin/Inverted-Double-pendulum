from typing import Tuple
from python.components import Action, StochasticDomain, State
from python.policy import Policy

import numpy as np 

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