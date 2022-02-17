from math import gamma
import numpy as np
from python.Q_learner import Q_function, Q_learn
from python.constants import *
from python.components import StochasticDomain
from python.policy import QLearningPolicy

if __name__ == '__main__':
    domain = StochasticDomain(G, W[0])
    qfunc = Q_learn(domain, GAMMA)
    print(qfunc)
    qlearning_policy = QLearningPolicy(domain, GAMMA)