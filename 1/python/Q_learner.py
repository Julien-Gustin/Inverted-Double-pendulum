from torch import q_per_channel_axis
from python.components import StochasticDomain, State
import numpy as np

def one_index(x: int, y: int, n: int):
    """ maps index (y, x) into y*n + x """
    return y*n+x

def two_indexes(index: int, n: int):
    y = index // n
    x = index % n
    return [x, y]

def Q_function(domain: StochasticDomain, decay: float, N: int):
    m, n = domain.g.shape
    nb_actions = len(domain.actions)
    nb_states = m*n

    Q_prec = None 
    Q_current = np.zeros((nb_states, nb_actions), dtype=float)

    for _ in range(N):
        Q_prec = Q_current.copy()
        for x in range(n):
            for y in range(m):
                state = State(x, y)
                for ai, action in enumerate(domain.actions):
                    reward_signal = domain.r(state, action)
                    recc_value = sum(domain.p(State(two_indexes(i, n)[0], two_indexes(i, n)[1]), state, action) * max(Q_prec[i, ]) for i in range(nb_states))
                    Q_current[one_index(x, y, n), ai] = reward_signal + decay*recc_value
    
    return Q_current


