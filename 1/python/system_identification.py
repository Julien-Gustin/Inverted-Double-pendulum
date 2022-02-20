import numpy as np

from python.components import action_to_index

class MDP:
    def __init__(self, state_space_x, state_space_y, action_space):
        self.N = np.zeros((state_space_x, state_space_y, action_space))
        self.N_2 = np.zeros((state_space_x, state_space_y, action_space, state_space_x, state_space_y))
        self.R = np.zeros((state_space_x, state_space_y, action_space))
        self.Q = np.zeros((state_space_x, state_space_y, action_space))


    def update(self, state, action, reward, next_state):
        action = action_to_index(action)

        self.N[state.x][state.y][action] += 1
        self.N_2[state.x][state.y][action][next_state.x][next_state.y] += 1
        self.R[state.x][state.y][action] += reward

    def r(self, state, action):
        miss = self.N == 0
        self.N[miss] = 1

        action = action_to_index(action)

        return (self.R/self.N)[state.x][state.y][action]

    def p(self, state, given_state, given_action):
        miss = self.N == 0
        self.N[miss] = 1

        given_action = action_to_index(given_action)

        return (self.N_2[given_state.x, given_state.y, given_action, state.x, state.y] / self.N[given_state.x, given_state.y, given_action])