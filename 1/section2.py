from pickle import TRUE
from section1 import *
import numpy as np
import math
import copy

def J(N:int, policy:Simulate, stochastic:bool):
    """ Approximation of the Expected return of a stationay policy for each state

    Args:
        N (int): Time horizon
        policy: (Simulate): Simulate the policy over a domain


    Returns:
        [np.array]: matrix of size n x m where the element [y, x] is 
                the approximation of the expected return of a stationary policy at state (x, y) 
    """ 
    m, n = G.shape
    j_prec = np.zeros((n, m))

    for _ in range(N):
    
        j_curr = np.zeros((n, m))

        for x in range(n):
            for y in range(m):

                initial_state = State(x, y)
                domain = Domain(initial_state, stochastic=stochastic) 
                simulation = policy(domain)

                if stochastic:
                    w0 = 0
                    for w in W:
                        simulation_copy = copy.deepcopy(simulation)
                        _ , _, r, x_new = simulation_copy.step(noise=w0)
                        j_curr[y, x] += (r + GAMMA * j_prec[x_new.y, x_new.x]) * w 
                        w0 += w

                else:
                    _ , _, r, x_new = simulation.step()
                    j_curr[y, x] = r + GAMMA * j_prec[x_new.y, x_new.x]  

        j_prec = j_curr 

    return j_curr

if __name__ == '__main__':
    epsilon = 1e-6
    N = math.ceil(math.log((epsilon * (1.0 -GAMMA))/ Br, GAMMA))
    print("N =", N)

    print("\n--- Deterministic ---\n")
    print(J(N, Simulate, stochastic=False))
    print("\n--- Stochastic ---\n")
    print(J(N, Simulate, stochastic=True))
