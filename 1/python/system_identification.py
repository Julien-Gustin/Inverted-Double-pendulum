import numpy as np

from python.components import action_to_index

class MDP:
    """Estimation of an mdp given a trajectory"""
    def __init__(self, state_space_x, state_space_y, action_space, trajectory):
        self.N = np.zeros((state_space_x, state_space_y, action_space, state_space_x, state_space_y))
        self.R = np.zeros((state_space_x, state_space_y, action_space))
        self.Q = np.zeros((state_space_x, state_space_y, action_space))

        for state, action, reward, next_state in trajectory:
            self._update(state, action, reward, next_state)

        self.state_space_x = state_space_x
        self.state_space_y = state_space_y


    def _update(self, state, action, reward, next_state):
        action = action_to_index(action)

        self.N[state.x][state.y][action][next_state.x][next_state.y] += 1
        self.R[state.x][state.y][action] += reward

    # These two functions could be optimize by using matrix form,
    # but it works quite well in that way
    def p(self, state, given_state, given_action):
        N_2 = self.N.sum(axis=(-1,-2))

        given_action = action_to_index(given_action)
        
        if N_2[given_state.x, given_state.y, given_action] == 0:
            return 1/(self.state_space_y * self.state_space_x)

        return (self.N[given_state.x, given_state.y, given_action, state.x, state.y] / N_2[given_state.x, given_state.y, given_action])

    def r(self, state, action):
        N_2 = self.N.sum(axis=(-1,-2))

        action = action_to_index(action)

        if self.R[state.x, state.y, action] == 0:
            return 0

        return (self.R[state.x][state.y][action]/N_2[state.x][state.y][action])