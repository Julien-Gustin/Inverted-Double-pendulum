import numpy as np

ACTIONS = [4, -4] #Action space discrete ? written as {-4, 4} and not [-4, 4] in the assignment

class State:
    def __init__(self, p: float, s: float):
        self.terminal = False
        if np.abs(p) > 1 or np.abs(s) > 3:
            self.terminal = True 
        self.p = p
        self.s = s

    @staticmethod
    def random_initial_state():
        p = np.random.uniform(-0.1, 0.1)
        s = 0
        return State(p, s)

    def is_terminal(self):
        return self.terminal

    def __repr__(self) -> str: 
        return "(p={}, s={})".format(self.p, self.s)

class CarOnTheHillDomain():
    def __init__(self, discount_factor=0.95, m=1, g=9.81, time_step=0.1, integration_time_step=0.001):
        self.discount_factor = discount_factor
        self.time_step = time_step
        self.integration_time_step = integration_time_step
        self.m = m
        self.g = g 

    def _Hill(self, p) -> float:
        return (p**2 + p) if p < 0 else p/(1+5*(p**2)**(0.5))

    def _Hill_first(self, p: float) -> float:
        return 2*p-1 if p < 0 else 1 / (1+5*p**2)**(1.5)

    def _Hill_second(self, p: float) -> float:
        return 2 if p < 0 else (-15*p)/(1+5*p**2)**(2.5)

    def f(self, state: State, action: int):
        if state.is_terminal():
            return state 

        p = state.p
        s = state.s 

        for _ in range(self.time_step // self.integration_time_step):
            hill_first = self._Hill_first(p)
            hill_second = self._Hill_second(p)
            s_first = action/(self.m*(1+hill_first**2)) - (self.g*hill_first + hill_first*hill_second*(s**2))/(1+hill_first**2)
            p += s*self.integration_time_step
            s += s_first*self.integration_time_step

        return State(p, s)
            

    def r(self, state: State, action: int):
        if state.is_terminal():
            return 0
        new_state = self.f(state, action)
        if new_state.p < -1 or np.abs(new_state.s) > 3:
            return -1

        return new_state.p > 1 and np.abs(new_state.s) <= 3


