from random import uniform
from python.components import StochasticDomain
from python.components import State

import numpy as np

from python.constants import GAMMA

class PolicySimulation():
    def __init__(self, domain: StochasticDomain, x_init: int, y_init: int):
        self.domain = domain
        self.state = State(x_init, y_init)

    #interface function
    def chooseAction(self):
        pass 

    def step(self):
        action = self.chooseAction()
        prev_state = self.state
        self.state, reward = self.domain.interact(self.state, action)
        return prev_state, action, reward, self.state

    def J(self, N: int):
        """
        Computes the expected reward for every state 
        of the domain if we follow N steps of this policy.
        """
        m, n = self.domain.g.shape
        j_prec = np.zeros((n, m))

        for _ in range(N):
            j_curr = np.zeros((n, m))
            for x in range(n):
                for y in range(m):
                    state = State(x, y)
                    action = self.chooseAction()
                    for transition in self.domain.possibleTransitions(state, action):
                        new_state, reward, probability = transition
                        j_curr[y, x] += probability*(reward + GAMMA*j_prec[new_state.y, new_state.x])

            j_prec = j_curr
        return j_curr

                    
class AlwaysGoRightPolicySimulation(PolicySimulation):
    def chooseAction(self):
        return self.domain.RIGHT