from argparse import Action
from sklearn.base import RegressorMixin, clone
from python.domain import ACTIONS, State
import numpy as np

class Fitted_Q():
    def __init__(self, model:RegressorMixin, stop, discount_factor:float) -> None:
        self.model = model

        if stop is None:
            exit() # TODO

        self.stop = stop
        self.discount_factor = discount_factor

    def fit(self, trajectories: list): # trajectories = [(x0, u0, r0, x1), (x10, u10, r10, x11)]
        """ Perform the fitted Q iteration given trajectories """

        X = trajectories[:, [0,1,2]] # [p, s, action]
        rewards = trajectories[:, 3]
        y = rewards 

        next_states = trajectories[:, [4, 5]]

        # generate combination of [state action] pairs
        X_1 = np.c_[np.repeat(next_states, 2, axis=0), np.tile(np.array(ACTIONS), len(next_states))] 

        terminal = rewards != 0
        Q_hat = None

        stop_condition = self.stop(Q_hat)
        while next(stop_condition):
            self.model.fit(X, y)
            Q_hat = self.model.predict(X_1).reshape(-1, 2)

            max_u = Q_hat.max(axis=1) # extract the  values doing the best actions

            # When a terminal state is reached, it can not gain anymore rewards afterward
            y = np.where(terminal, rewards, self.discount_factor * max_u)

    def predict(self, X):
        """ predict given state action pairs """
        return self.model.predict(X)

    def compute_optimal_actions(self, states):
        """ Return the optimal action to perform in a given state """
        X = np.c_[np.repeat(states, len(ACTIONS), axis=0), np.tile(np.array(ACTIONS), len(states))]

        y_pred = self.predict(X)
        y_pred = y_pred.reshape(-1, 2)
        y_pred = np.argmax(y_pred, axis=1)

        optimal_action = np.array([ACTIONS[index] for index in y_pred])
        if len(optimal_action) == 1:
            return optimal_action[0]
            
        return optimal_action