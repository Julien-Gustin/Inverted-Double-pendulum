from python.constants import G, GAMMA, W
from python.policy import RandomUniformPolicy, TrajectoryBasedQLearningPolicy, Simulation
from python.components import DeterministicDomain, StochasticDomain, State

if __name__ == "__main__":
    policy = RandomUniformPolicy()
    domain = StochasticDomain(G, W[0])
    #domain = DeterministicDomain(G)
    initial_state = State(0, 3)
    simulation = Simulation(domain, policy, initial_state, seed=42)

    trajectory_steps = 10**6
    learning_ratio = 0.05

    trajectory_based_policy = TrajectoryBasedQLearningPolicy(domain, initial_state, trajectory_steps, learning_ratio, GAMMA, seed=2)
    print(trajectory_based_policy.Q_policy.T)
    print(trajectory_based_policy.J(domain))