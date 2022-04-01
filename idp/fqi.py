import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import math

B_r = 10 # Max rewards for double pendulum environment

class Fitted_Q_ERT():
    def __init__(self, discount_factor:float, actions:list, seed=42) -> None:
        self.model = None
        self.get_model = lambda : ExtraTreesRegressor(n_estimators=50, random_state=seed, n_jobs=-1)
        self.actions = actions
        self.discount_factor = discount_factor

    def fit(self, trajectories: list):
        """ Perform the fitted Q iteration given trajectories """
        epsilon = 1e-1
        N = math.ceil(math.log((epsilon / (2 * B_r)) * (1. - self.discount_factor) ** 2, self.discount_factor)) # bound

        columns = list(zip(*trajectories))
        state = np.array(columns[0]) 
        action = np.array(columns[1])
        rewards = np.array(columns[2])
        next_states = np.array(columns[3])
        done = np.array(columns[4])

        X = np.c_[state, action]
        y = rewards     

        # generate combination of [state action] pairs
        X_next = np.c_[np.repeat(next_states, len(self.actions), axis=0), np.tile(np.array(self.actions), len(next_states))] 

        for n in range(N):
            model = self.get_model()
            model.fit(X, y)

            Q_hat = model.predict(X_next).reshape(-1, len(self.actions))

            max_u = Q_hat.max(axis=1) # extract the values doing the best actions

            # When a terminal state is reached, it can not gain anymore rewards afterward
            y = np.where(done, rewards, rewards + self.discount_factor * max_u)

            print("\r", n+1, "/", N, end="\r")

        self.model = model # last model, Q_N

    def predict(self, X):
        """ predict given state action pairs """
        return self.model.predict(X)

    def compute_optimal_actions(self, states):
        """ Return the optimal action to perform in a given state """

        # generate combination of [state action] pairs
        X = np.c_[np.repeat(states, len(self.actions), axis=0), np.tile(np.array(self.actions), len(states))]

        y_pred = self.predict(X)
        y_pred = y_pred.reshape(-1, len(self.actions))
        y_pred = np.argmax(y_pred, axis=1)

        optimal_action = np.array([self.actions[index] for index in y_pred])
        if len(optimal_action) == 1:
            return optimal_action[0]
            
        return optimal_action