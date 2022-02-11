import numpy as np
from random import uniform
from random import randrange
from pprint import pprint

# Instance of the domain
G = np.array(
            [
                [-3, 1, -5, 0, 19],
                [6, 3, 8, 9, 10],
                [5, -8, 4, 1, -8],
                [6, -9, 4, 19, -5],
                [-20, -17, -4, -4, 9]
            ]
        )

# Bound on the reward
Br = G.max() 

# Decay factor
GAMMA = 0.99

# Probability distribution
W = [0.5, 0.5]

class Action:
    def __init__(self, action:tuple) -> None:
        self.i = action[1]
        self.j = action[0]

    def __repr__(self):
        return "(" + str(self.i) + "," + str(self.j) + ")"

class State:
    """ State space """
    def __init__(self, x, y) -> None:
        m, n = G.shape
        if x < 0 or x >= n or y < 0 or y >= m:
            print("ERROR\n\t `0 <= x < n` and `0 <= y < m`")
            exit(0)
        
        self.x = x
        self.y = y

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

class Domain:
    """ Domain instance """
    
    # Action space
    LEFT = Action((0, -1))
    RIGHT = Action((0, 1))
    UP = Action((-1, 0))
    DOWN = Action((1, 0))
    
    def __init__(self, initial_state: State, stochastic:bool=False) -> None:
        self.g = G
        self.m, self.n = G.shape
        self.state = initial_state
        self.stochastic=stochastic

    def F(self, action: Action):
        """ Update the state deterministically """
        new_x = min(max(self.state.x + action.i, 0), self.n - 1)
        new_y = min(max(self.state.y + action.j, 0), self.m - 1)
        
        new_state = State(new_x, new_y)
        self.state = new_state

        return new_state

    def f(self, action: Action, noise:float=0.5):
        """ Dynamics functions """
        
        if noise < 0.5 or not self.stochastic:
            self.F(action)

        else:
            self.state = State(0, 0)

        return self.state


    def R(self):
        return self.g[self.state.y][self.state.x]

    def r(self, action:Action, noise:float):
        """ Reward signal """
        self.f(action, noise)
        return self.R()


    def print_domain(self):
        """ Clean print of the domain """
        tab = self.g
        res = [list(map(str, sub)) for sub in tab]
        res[self.state.y][self.state.x] =  ">" + res[self.state.y][self.state.x] + "<"
        pprint(res)


class Simulate():
    """ Simulation of a policy """
    def __init__(self, environment: Domain) -> None:
        self.environment = environment

    def step(self, noise=None):
        """ Always go to right policy"""

        prev_state = self.environment.state
        action = self.environment.RIGHT

        if noise is None:
            noise = uniform(0, 1)

        reward = self.environment.r(action, noise)

        return prev_state, action, reward, self.environment.state



if __name__ == '__main__':
    initial_state = State(0, 3)
    domain = Domain(initial_state, stochastic=False)
    simulation = Simulate(domain)

    for t in range(10):
        simulation.environment.print_domain()
        print(simulation.step())

    print("\n", "-"*50, "\n")

    initial_state = State(0, 3)
    domain = Domain(initial_state, stochastic=True)
    simulation = Simulate(domain)

    for t in range(10):
        simulation.environment.print_domain()
        print(simulation.step())