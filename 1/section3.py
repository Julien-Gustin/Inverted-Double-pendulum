import math 

from python.Q_learner import Q_learn
from python.constants import *
from python.components import StochasticDomain
from python.policy import QLearningPolicy

if __name__ == '__main__':
    epsilon = 1e-6
    N = math.ceil(math.log((epsilon * (1.0 -GAMMA))/ Br, GAMMA))
    domain = StochasticDomain(G, W[0])
    qlearning_policy = QLearningPolicy(domain, GAMMA)
    #print(qlearning_policy.J(domain, GAMMA, N))
    print(qlearning_policy.Q_policy)