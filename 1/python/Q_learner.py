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

    return Q_current

def Q_learn_temporal_difference(state_space, action_space, trajectory, learning_rate=0.05, decay=0.99):
    n, m = state_space.shape
    nb_actions = len(action_space)
    Q_table = np.zeros((n, m, nb_actions), dtype=float)

    for t in trajectory: 
        #one-step transition of our trajectory
        starting_state, u, r, new_state = t
        u = action_space.index(u)
        #update rule
        Q_table[starting_state.x, starting_state.y, u] += learning_rate*(r + decay*max(Q_table[new_state.x, new_state.y, ]) 
                                                                            - Q_table[starting_state.x, starting_state.y, u])
    return Q_table






