from python.components import StochasticDomain, State
from python.policy import Policy

import numpy as np 

class Simulation():
    def __init__(self, domain: StochasticDomain, policy: Policy, x_init: int, y_init: int):
        self.domain = domain
        self.state = State(x_init, y_init)
        self.policy = policy 

    def step(self):
        action = self.policy.chooseAction(self.state)
        prev_state = self.state
        self.state, reward = self.domain.interact(self.state, action)
        return prev_state, action, reward, self.state