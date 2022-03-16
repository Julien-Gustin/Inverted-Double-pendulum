import numpy as np

from python.domain import CarOnTheHillDomain, State
from python.policy import Policy

class Simulation():
    """
        domain: the parametrized car on the hill domain
        policy: the policy used on the domain
        remember_trajectory: if true, it keeps track of the trajectory done by the simulator
        initial_state: initial state on the domain
    """
    def __init__(self, domain: CarOnTheHillDomain, policy: Policy, initial_state: State, remember_trajectory=False, seed=43, stop_when_terminal=True) -> None:
        self.domain = domain
        self.policy = policy
        self.state = initial_state
        self.stop_when_terminal = stop_when_terminal

        self.trajectory = None 
        if remember_trajectory:
            self.trajectory = list()

        np.random.seed(seed)

    def step(self, values=False):
        """
        Simulate one step of the policy
        """
        action = self.policy.make_action(self.state)
        previous_state = self.state
        self.state = self.domain.f(self.state, action)
        reward = self.domain.r(previous_state, action) 

        if values:
            return (*previous_state.values(), action, reward, *self.state.values())

        return previous_state, action, reward, self.state

    def simulate(self, steps: int) -> None:
        """
        Simulate (steps) steps on the domain, according to the policy
        If remember_trajectory=True, each transition will be remembered
        """
        for _ in range(steps):
            if self.stop_when_terminal and self.state.is_terminal():
                return
            previous_state, action, reward, _ = self.step()

            if self.trajectory is not None:
                self.trajectory.append((previous_state, action, reward, self.state))

    def get_trajectory(self, values=False) -> list:
        if values:
            trajectory = [[*state.values(), action, reward, *next_state.values()] for state, action, reward, next_state in self.trajectory]
            return trajectory

        return self.trajectory


