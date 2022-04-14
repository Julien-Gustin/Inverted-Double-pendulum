import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import math
from copy import deepcopy
from models.utils.expected_return import J

import matplotlib.pyplot as plt

B_r = 10 # Max rewards for double pendulum environment

class Fitted_Q_ERT():
    def __init__(self, discount_factor:float, actions:list, env, seed=42) -> None:
        epsilon = 1e-1

        self.model = None
        self.get_model = lambda : ExtraTreesRegressor(n_estimators=50, random_state=seed, n_jobs=-1)
        self.actions = actions
        self.discount_factor = discount_factor
        self.env = env
        self.J_mean = []
        self.J_std = []

        N = math.ceil(math.log((epsilon / (2 * B_r)) * (1. - self.discount_factor) ** 2, self.discount_factor)) # bound
        self.N = min(N, 200) # higher than 200 is useless and computational

    def fit(self, trajectories: list, compute_j:bool):
        """ Perform the fitted Q iteration given trajectories """
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

        for n in range(self.N):
            model = self.get_model()
            model.fit(X, y)

            Q_hat = model.predict(X_next).reshape(-1, len(self.actions))

            max_u = Q_hat.max(axis=1) # extract the values doing the best actions

            # When a terminal state is reached, it can not gain anymore rewards afterward
            y = np.where(done, rewards, rewards + self.discount_factor * max_u)

            print("\r", n+1, "/", self.N, end="\r")
            self.model = model
            if compute_j:
                js= self.compute_j(n)
                self.J_mean.append(js[0])
                self.J_std.append(js[1])


    def predict(self, X):
        """ predict given state action pairs """
        return self.model.predict(X)

    def compute_optimal_actions(self, states):
        """ Return the optimal action to perform in a given state """

        # generate combination of [state action] pairs
        states = np.array([states])
        X = np.c_[np.repeat(states, len(self.actions), axis=0), np.tile(np.array(self.actions), len(states))]
        y_pred = self.predict(X)
        y_pred = y_pred.reshape(-1, len(self.actions))
        y_pred = np.argmax(y_pred, axis=1)

        optimal_action = np.array([self.actions[index] for index in y_pred])
        if len(optimal_action) == 1:
            return optimal_action[0]
            
        return optimal_action

    def compute_j(self, n):
        mean, std = J(self.env, self,self.discount_factor, 20, 1000)
        print("N = {} | mean: {} | std: {}".format(n+1, mean, std))
        return mean, std

    def make_plot(self):
        J_mean = np.array(self.J_mean)
        J_std = np.array(self.J_std)
        plt.plot(J_mean, label="Expected return")
        plt.ylabel("Expected return J")
        plt.xlabel("N")
        plt.legend()
        plt.fill_between(range(self.N),J_mean-J_std,J_mean+J_std,alpha=.1)
        plt.savefig("figures/J_fqi.png")
