from torch import nn
from python.domain import *
import torch
import numpy as np
from python.constants import *
from python.simulation import Simulation
import torch.utils.data as data
from python.policy import  EpsilonGreedyPolicy

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
        self.model.eval()
        with torch.no_grad():
            out_data = self.model(torch.Tensor(X)).detach().numpy()
            
        return out_data

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


class OnlineParametricQLearning(ParametricQLearning):
    def __init__(self, model, domain) -> None:
        super().__init__(model)
        self.policy = EpsilonGreedyPolicy(self, 0.9, seed=42)
        domain = CarOnTheHillDomain(DISCOUNT_FACTOR, M, GRAVITY, TIME_STEP, INTEGRATION_TIME_STEP)
        self.domain = domain
        
    def fit(self, N):
        n = 0
        trajectories = []

        simulator = Simulation(self.domain, self.policy, State.random_initial_state(seed=0), seed=42)
        while n < N:
            if simulator.state.is_terminal():
                simulator = Simulation(self.domain, self.policy, State.random_initial_state(seed=n), seed=n)

            prec_state = simulator.state
            trajectories.append([prec_state, None, None, None])

            p_prec, s_prec, action, reward, p, s = simulator.step(values=True)

            pred_Q_t = self.model(torch.Tensor([[p_prec, s_prec, action]]))

            with torch.no_grad():
                pred_Q_t_1 = self.model(torch.Tensor([[p, s, ACTIONS[0]], [p, s, ACTIONS[1]]]))

            print("\r", n , "/", N, end="\r")

            self.optimizer.zero_grad()
            loss = self._loss(pred_Q_t, pred_Q_t_1, torch.Tensor([reward]))
            loss.backward()

            ## Gradient norm, https://jermwatt.github.io/machine_learning_refined/notes/3_First_order_methods/3_9_Normalized.html

            # norm = torch.norm


            norm = torch.norm(torch.stack([
                torch.norm(p.grad, 2.)
                for p in self.model.parameters()
            ]), 2.)

            ## Normalize
            for p in self.model.parameters():
                p.grad /= norm + 1e-7

            self.optimizer.step()

            n += 1

        return trajectories