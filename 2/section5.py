from section4 import plot_mu, plot_Q
from python.policy import ParameterQPolicy
from section3 import make_video
from python.constants import *
from python.simulation import Simulation
import matplotlib.pyplot as plt
from python.constants import *
from python.domain import State, CarOnTheHillDomain
from python.simulation import Simulation
import torch.utils.data as data
from python.expected_return import J
from python.policy import RandomActionPolicy
from python.parametric_Q_learning import *

def get_trajectories(buffer_size):
    # Random
    domain = CarOnTheHillDomain()
    random_policy = RandomActionPolicy()
    random_trajectories = np.array([])

    random_trajectories = []
    
    trajectory_length = 1000
    n = 0
    while len(random_trajectories) <= buffer_size:
        initial_state = State.random_initial_state(seed=n)
        simulation = Simulation(domain, random_policy, initial_state, remember_trajectory=True, seed=n, stop_when_terminal=True)
        simulation.simulate(trajectory_length)
        random_trajectories.extend(np.array(simulation.get_trajectory(values=True)).squeeze())
        print("\r ", len(random_trajectories), "/", buffer_size, end="\r")
        n += 1

    random_trajectories = np.array(random_trajectories[:buffer_size]) 

    return random_trajectories


if __name__ == '__main__':
    
    # Generate dataset
    trajectories = get_trajectories(50000) # random trajectories
    dataset = StateActionDataset(trajectories)
    loader = data.DataLoader(
        dataset,
        batch_size=None,
        sampler=data.BatchSampler(data.RandomSampler(dataset), 32, False)
    )
    
    # Learn
    pql = ParametricQLearning(net)
    loss = pql.fit(loader, 500)

    # Make plot
    plot_Q(pql, "PQL")
    plot_mu(pql, "PQL")

    # generate video
    initial_state = State.random_initial_state()
    policy = ParameterQPolicy(pql)
    domain = CarOnTheHillDomain(DISCOUNT_FACTOR, M, GRAVITY, TIME_STEP, INTEGRATION_TIME_STEP)
    simulation = Simulation(domain, policy, initial_state, remember_trajectory=True, seed=42)
    simulation.simulate(600)
    make_video("videos/animation_parameter_Q.avi", simulation.get_trajectory())

    j = J(domain, policy, DISCOUNT_FACTOR, 50, 200)
    print(j)