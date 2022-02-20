from python.components import StochasticDomain, State
import numpy as np

def Q_function(domain: StochasticDomain, decay: float, Q_prec: np.array):
    m, n = domain.g.shape
    Q_current = np.zeros(Q_prec.shape, dtype=float)
    for x in range(n):
        for y in range(m):
            state = State(x, y)
            for ai, action in enumerate(domain.actions):
                reward_signal = domain.r(state, action)
                recc_value = sum(domain.p(State(i, j), state, action) 
                                * max(Q_prec[i,j, ]) for i in range(n) for j in range(m))

                Q_current[x, y, ai] = reward_signal + decay*recc_value
    
    return Q_current


def Q_learn(domain: StochasticDomain, decay: float, N):
    n, m = domain.g.shape
    nb_actions = len(domain.actions)
    Q_current = np.zeros((n, m, nb_actions), dtype=float)
    
    for _ in range(N):
        Q_current = Q_function(domain, decay, Q_current)
    
    return Q_current
        

def Q_function_estimation(domain, decay, Q_prec, mdp):
    m, n = domain.g.shape
    Q_current = np.zeros(Q_prec.shape, dtype=float)
    for x in range(n):
        for y in range(m):
            state = State(x, y)
            for ai, action in enumerate(domain.actions):
                reward_signal = mdp.r(state, action)
                recc_value = sum(mdp.p(State(i, j), state, action) 
                                * max(Q_prec[i,j, ]) for i in range(n) for j in range(m))

                Q_current[x, y, ai] = reward_signal + decay*recc_value

    return Q_current


def Q_learn_estimation(domain, decay, N, mdp):
    n, m = domain.g.shape
    nb_actions = len(domain.actions)
    Q_current = np.zeros((n, m, nb_actions), dtype=float)

    for _ in range(N):
        Q_current = Q_function_estimation(domain, decay, Q_current, mdp)

    # current_action_indexes = np.array([domain.actions[np.argmax(Q_current[i,j, ])] for i in range(n) for j in range(m)]).reshape(n, m)
    return Q_current






