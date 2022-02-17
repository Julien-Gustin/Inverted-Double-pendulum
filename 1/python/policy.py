from python.components import Action
from python.components import State, StochasticDomain

import numpy as np

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
                        j_curr[y, x] += probability*(reward + decay*j_prec[new_state.y, new_state.x])

            j_prec = j_curr
        return j_curr

class AlwaysGoRightPolicy(Policy):
    def chooseAction(self, state: State):
        return Action((0, 1))