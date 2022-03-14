from torch import detach, nn
from python.domain import ACTIONS
import torch
import numpy as np
from python.constants import *
from python.simulation import Simulation
import torch.utils.data as data
from python.expected_return import J
import torch.nn.functional as F

net = nn.Sequential(
    nn.Linear(3, 5),
    nn.ReLU(),
    nn.Linear(5, 5),
    nn.ReLU(),
    nn.Linear(5, 5),
    nn.ReLU(),
    nn.Linear(5, 1), 
)

torch.manual_seed(0)


class StateActionDataset(data.Dataset):
    def __init__(self, trajectories):
        X = torch.Tensor(trajectories[:, [0,1,2]]) # [p, s, action]
        rewards = torch.Tensor(trajectories[:, 3])

        next_states = torch.Tensor(trajectories[:, [4, 5]])
        X_1 = torch.Tensor(np.c_[np.repeat(next_states, 2, axis=0), np.tile(np.array(ACTIONS), len(next_states))])
        X_1 = X_1.reshape((len(X_1)//2, 6))

        self.X, self.reward = X, rewards
        self.X_prime = X_1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i: int):
        return self.X[i], self.reward[i], self.X_prime[i]


class ParametricQLearning():
    def __init__(self, model) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.gamma = 0.95


    def _loss(self, pred_Q_t, pred_Q_t_1, rewards):
        terminal = rewards != 0
        with torch.no_grad():
            max_pred_Q_t_1 = pred_Q_t_1.reshape(-1, 2).max(dim=1)[0].view(-1)
            max_pred_Q_t_1_corrected = torch.where(terminal, rewards, self.gamma * max_pred_Q_t_1)
            delta = pred_Q_t - max_pred_Q_t_1_corrected
        
        return (delta * pred_Q_t).mean()


    def fit(self, loader:StateActionDataset, nb_epoch):

        train_loss = []

        for epoch in range(nb_epoch):
            losses = []
            for X, rewards, X_1 in loader:
                X_1 = X_1.reshape((len(X_1) * 2, 3))
                pred_Q_t = self.model(X)
                with torch.no_grad():
                    pred_Q_t_1 = self.model(X_1)

                self.optimizer.zero_grad()
                loss = self._loss(pred_Q_t, pred_Q_t_1, rewards)
                loss.backward()
                self.optimizer.step()

                losses.append(loss.detach().numpy())

            train_loss.append(np.array(losses).mean())
            print("\r", epoch,  "/", nb_epoch, end="\r")

        return train_loss

    def predict(self, X):
        return self.model(torch.Tensor(X)).detach().numpy()

    def compute_optimal_actions(self, states):
        """ Return the optimal action to perform in a given state """
        X = torch.Tensor(np.c_[np.repeat(states, len(ACTIONS), axis=0), np.tile(np.array(ACTIONS), len(states))])

        y_pred = self.predict(X)
        y_pred = y_pred.reshape(-1, 2)
        y_pred = np.argmax(y_pred, axis=1)

        optimal_action = np.array([ACTIONS[index] for index in y_pred])
        if len(optimal_action) == 1:
            return optimal_action[0]
            
        return optimal_action