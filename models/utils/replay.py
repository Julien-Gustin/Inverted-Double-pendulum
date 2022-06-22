import numpy as np
import torch 

np.random.seed(42)

class ReplayBuffer():
    """ Cyclic replay buffer of a given capactity """
    def __init__(self, capacity:int):
        self.capacity = capacity
        self.index = 0
        self.states = np.zeros((capacity, 9))
        self.actions = np.zeros(capacity)
        self.rewards = np.zeros(capacity)
        self.new_states = np.zeros((capacity, 9))
        self.done = np.zeros(capacity)

        self.size = 0

    def __len__(self):
        return self.size

    def store(self, sample:list):
        """ store a new sample to the replay buffer"""
        state, action, reward, new_state, done = sample

        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.new_states[self.index] = new_state
        self.done[self.index] = done 

        self.index = (self.index+1) % self.capacity

        if self.size != self.capacity:
            self.size += 1

    def minibatch(self, size:int):
        """ Return minibatch of size `size` """
        indexes = np.random.randint(0, self.size, size=size)

        batch_dic = {
                    'states': torch.Tensor(self.states[indexes]),
                    'actions': torch.Tensor(self.actions[indexes]).unsqueeze(1),
                    'rewards': torch.Tensor(self.rewards[indexes]).unsqueeze(1),
                    'new_states': torch.Tensor(self.new_states[indexes]),
                    'done': torch.Tensor(self.done[indexes]).unsqueeze(1)
        }
        return batch_dic