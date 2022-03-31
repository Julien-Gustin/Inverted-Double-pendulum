import numpy as np

class OU():
    def __init__(self, initial_x, mu, theta, sigma):
        self.x = initial_x
        self.mu = mu
        self.theta = theta
        self.sigma = sigma 
    
    def __call__(self):
        #Formula of an OU noise 
        self.x = self.theta*(self.mu-self.x) + self.sigma*np.random.randn(1)
        return self.x 