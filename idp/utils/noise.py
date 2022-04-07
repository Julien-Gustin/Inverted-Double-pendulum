import numpy as np

class OU():
    """ Implementation of an Ornstein-Uhlenbeck process """
    def __init__(self, initial_x, mu:float, theta:float, sigma:float):
        self.x = initial_x
        self.mu = mu
        self.theta = theta
        self.sigma = sigma 
    
    def __call__(self):
        self.x = self.theta*(self.mu-self.x) + self.sigma*np.random.randn(1)
        return self.x 

class Gaussian():
    """ Gaussian noise """
    def __init__(self, sigma):
        self.sigma = sigma 
    
    def __call__(self):
        return self.sigma*np.random.randn()