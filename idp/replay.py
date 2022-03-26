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

    def store(self, sample):
        state, action, reward, new_state, done = sample
        self.states = np.append(self.states, state)
        self.actions = np.append(self.actions, action)
        self.rewards = np.append(self.rewards, reward)
        self.new_states = np.append(self.new_states, new_state)
        self.done = np.append(self.done, done)

        if len(self.states == self.capacity+1):
            self.states = np.delete(self.states, 0)
            self.actions = np.delete(self.actions, 0)
            self.rewards = np.delete(self.rewards, 0)
            self.new_states = np.delete(self.new_states, 0)
            self.done = np.append(self.done, 0)

    def minibatch(self, size):
        indexes = None 
        if size > len(self.states):
            indexes = np.array([i for i in range(len(self.states))])
        else:
            indexes = np.random.randint(0, len(self.states), size=size)
        batch_dic = {
                    'states': torch.as_tensor(self.states[indexes]),
                    'actions': torch.as_tensor(self.actions[indexes]),
                    'rewards': torch.as_tensor(self.rewards[indexes]),
                    'new_states': torch.as_tensor(self.new_states[indexes]),
                    'done': torch.as_tensor(self.done[indexes])
        }
        return batch_dic