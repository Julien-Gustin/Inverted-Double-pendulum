import numpy as np
from pprint import pprint
from python.constants import *

class Action:
    def __init__(self, action:tuple) -> None:
        self.i = action[0]
        self.j = action[1]

    def __repr__(self):
        if self.i == -1 and self.j == 0:
            return "LEFT"
        elif self.i == 1 and self.j == 0:
            return "RIGHT"
        elif self.i == 0 and self.j == 1:
            return "DOWN"
        elif self.i == 0 and self.j == -1:
            return "UP"

LEFT = Action((-1, 0))
RIGHT = Action((1, 0))
UP = Action((0, -1))
DOWN = Action((0, 1))


def action_to_index(action):
    if action == LEFT:
        return 0
    
    if action == RIGHT:
        return 1

    if action == UP:
        return 2

    return 3


class State:
    """ State space """
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

    def __eq__(self, other):
        if isinstance(other, State):
            return self.x == other.x and self.y == other.y
        return False 

class StochasticDomain(): 
    # Action space
    actions = [LEFT, RIGHT, UP, DOWN] #iterative purposes

    def __init__(self, g, w) -> None:
        self.g = g
        self.w = w
        self.m, self.n = g.shape

    def interact(self, state: State, action: Action):
        """Interacts with the domain, returns a (state, reward) pair"""
        noise = np.random.rand()
        new_state = self.f(state, action, noise)
        reward = self.R(new_state)
        return (new_state, reward)

    def f(self, state: State, action: Action, noise: float):
        """ Dynamics functions """
        new_state = None
        if noise <= self.w:
            new_state = self.F(state, action)
        else:
            new_state = self.state = State(0, 0)
        return new_state

    def r(self, state: State, action: Action, noise: float):
        """ Reward signal """
        new_state = self.f(state, action, noise)
        return self.R(new_state)

    def r(self, state: State, action: Action):
        """ Expected reward """
        return self.w * self.R(self.F(state, action)) + (1-self.w) * self.R(State(0, 0))
    
    def p(self, state: State, given_state: State, given_action: Action):
        """ Probability of reaching state with given_state and given_action """
        possible_transitions = self.possibleTransitions(given_state, given_action)
        acc = 0
        for transition in possible_transitions:
            new_state, _, probability = transition
            if state == new_state:
                acc += probability

        return acc

    def possibleTransitions(self, state: State, action: Action):
        """
        Returns a list of (new_state, reward, probability) tuples 
        for each possible transitions of this action in the current state.
        """
        first_state = self.F(state, action)
        first_reward = self.R(first_state)

        second_state = State(0, 0)
        second_reward = self.R(second_state)

        return [(first_state, first_reward, self.w), (second_state, second_reward, 1-self.w)]

    def F(self, state: State, action: Action):
        """ Update the state deterministically """
        new_x = min(max(state.x + action.i, 0), self.n - 1)
        new_y = min(max(state.y + action.j, 0), self.m - 1)

        new_state = State(new_x, new_y)
        return new_state

    def R(self, state: State):
        return self.g[state.x][state.y]

    def print_domain(self, state):
        """ Clean print of the domain """
        tab = self.g
        res = [list(map(str, sub)) for sub in tab]
        res[state.x][state.y] =  ">" + res[state.x][state.y] + "<"
        pprint(res)

class DeterministicDomain(StochasticDomain):
    def __init__(self, g) -> None:
        super().__init__(g, 1.0)



