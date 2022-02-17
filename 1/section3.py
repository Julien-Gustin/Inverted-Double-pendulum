import numpy as np
from python.Q_learner import Q_function
from python.constants import *
from python.components import StochasticDomain

if __name__ == '__main__':
    domain = StochasticDomain(G, W[0])
    N = 1000
    Q_n = Q_function(domain, GAMMA, N)
    print(Q_n)