from types import new_class
from matplotlib.pyplot import axis
import numpy as np
import torch 

class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.states = np.array(list())
        self.actions = np.array(list())
        self.rewards = np.array(list())
        self.new_states = np.array(list())
        self.done = np.array(list())

    def __len__(self):
        return len(self.states)

    def store(self, sample):
        state, action, reward, new_state, done = sample

        self.states = self.states.tolist()
        self.states.append(state)
        self.states = np.array(self.states)

        self.actions = np.append(self.actions, action)
        self.rewards = np.append(self.rewards, reward)

        self.new_states = self.new_states.tolist()
        self.new_states.append(new_state)
        self.new_states = np.array(self.new_states)

        self.done = np.append(self.done, done)

        if len(self.states) == self.capacity+1:
            self.states = np.delete(self.states, 0, axis=0)
            self.actions = np.delete(self.actions, 0)
            self.rewards = np.delete(self.rewards, 0)
            self.new_states = np.delete(self.new_states, 0, axis=0)
            self.done = np.delete(self.done, 0)

    def minibatch(self, size):
        indexes = np.random.randint(0, len(self.states), size=size)

        batch_dic = {
                    'states': torch.Tensor(self.states[indexes]),
                    'actions': torch.Tensor(self.actions[indexes]).unsqueeze(1),
                    'rewards': torch.Tensor(self.rewards[indexes]).unsqueeze(1),
                    'new_states': torch.Tensor(self.new_states[indexes]),
                    'done': torch.Tensor(self.done[indexes]).unsqueeze(1)
        }
        return batch_dic