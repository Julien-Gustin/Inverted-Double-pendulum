from pyparsing import null_debug_action
from torch import q_per_channel_axis
from python.components import StochasticDomain, State
import numpy as np
import math 

def one_index(x: int, y: int, n: int):
    """ maps index (y, x) into y*n + x """
    return y*n+x

def two_indexes(index: int, n: int):
    y = index // n
    x = index % n
    return [x, y]

def Q_function(domain: StochasticDomain, decay: float, Q_prec: np.array):
    m, n = domain.g.shape
    nb_states = m*n

    Q_current = np.zeros(Q_prec.shape, dtype=float)
    for x in range(n):
        for y in range(m):
            state = State(x, y)
            for ai, action in enumerate(domain.actions):
                reward_signal = domain.r(state, action)
                recc_value = sum(domain.p(State(two_indexes(i, n)[0], two_indexes(i, n)[1]), state, action) 
                                * max(Q_prec[i, ]) for i in range(nb_states))
                Q_current[one_index(x, y, n), ai] = reward_signal + decay*recc_value
    
    return Q_current


def Q_learn(domain: StochasticDomain, decay: float):
    n, m = domain.g.shape
    nb_states = n*m
    nb_actions = len(domain.actions)

    Q_current = np.zeros((nb_states, nb_actions), dtype=float)
    prec_action_indexes = [0]*nb_states
    N = 0

    while True:
        Q_current = Q_function(domain, decay, Q_current)
        current_action_indexes = [np.argmax(Q_current[i, ]) for i in range(nb_states)]
        if prec_action_indexes == current_action_indexes:
            print("Found in {} iterations".format(N))
            return np.array([domain.actions[ai] for ai in current_action_indexes]).reshape(n, m)
        
        prec_action_indexes = current_action_indexes
        N+=1







