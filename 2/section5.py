from python.simulation import Simulation
import matplotlib.pyplot as plt
from python.constants import *
from python.domain import State, CarOnTheHillDomain
from python.simulation import Simulation
import torch.utils.data as data
from python.expected_return import J
from python.policy import RandomActionPolicy
from python.parametric_Q_learning import *
import math 
from python.fitted_Q import Fitted_Q
from section4 import plot_mu, plot_Q, bound_stop
from python.policy import FittedQPolicy, ParameterQPolicy
from python.neural_network import NN
import math 
import matplotlib

def get_net():
    net = nn.Sequential(
        nn.Linear(3, 5),
        nn.ReLU(),
        nn.Linear(5, 5),
        nn.ReLU(),
        nn.Linear(5, 5),
        nn.ReLU(),
        nn.Linear(5, 1), 
    )
    return net

def curve_plot(expected_FQI, expected_PQ, expected_online, trajectory_sizes):
    print("FQI")
    print(expected_FQI)
    print("PQ")
    print(expected_PQ)
    print("PQ-Online")
    print(expected_online)
    fig1, ax1 = plt.subplots()
    ax1.plot(trajectory_sizes, expected_FQI, label="FQI")
    ax1.plot(trajectory_sizes, expected_PQ, label="PQ")
    ax1.plot(trajectory_sizes, expected_online,label="PQ-Online")
    ax1.legend()
    plt.title("Expected return of the algorithm with respect to the trajectory size")
    plt.xticks(trajectory_sizes)
    plt.xscale("log")
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.ylabel("Expected return J")
    plt.xlabel("Trajectory size")
    plt.savefig("figures/comparison.png")

def get_trajectories(buffer_size, seed=42):
    # Random
    domain = CarOnTheHillDomain()
    random_policy = RandomActionPolicy(seed)
    random_trajectories = np.array([])

    random_trajectories = []
    
    trajectory_length = 1000
    n = 0
    terminal = []
    while len(random_trajectories) <=  buffer_size:
        initial_state = State.random_initial_state(seed=seed + n*buffer_size)
        simulation = Simulation(domain, random_policy, initial_state, remember_trajectory=True, seed=seed + n*buffer_size, stop_when_terminal=True)
        simulation.simulate(trajectory_length)
        terminal.append(np.array(simulation.get_trajectory(values=True))[:, 3][-1])
        random_trajectories.extend(np.array(simulation.get_trajectory(values=True)).squeeze())
        n += 1

    terminal = np.array(terminal)

    random_trajectories = np.array(random_trajectories[:buffer_size]) 

    return random_trajectories

def protocol_comparison(online=False):
    trajectory_sizes = [5000, 10000, 20000, 50000, 100000, 500000]
    domain = CarOnTheHillDomain(DISCOUNT_FACTOR, M, GRAVITY, TIME_STEP, INTEGRATION_TIME_STEP)

    N_nn = math.ceil(math.log((1e-2 / (2 * B_r)) * (1. - DISCOUNT_FACTOR) ** 2, DISCOUNT_FACTOR))
    N_j = math.ceil(math.log((1e-3 / B_r) * (1. - DISCOUNT_FACTOR), DISCOUNT_FACTOR))

    expected_FQI = []
    expected_PQ = []
    expected_online = []

    for trajectory_size in trajectory_sizes:
        trajectories = get_trajectories(trajectory_size, seed=trajectory_size)

        # Q Fitted iteration
        mlp = lambda : NN(layers=2, neurons=5, output=1, epochs=5, batch_size=32, activation="relu")
        fqi = Fitted_Q(mlp, bound_stop, DISCOUNT_FACTOR)

        fqi.fit(trajectories)
        fqi_policy = FittedQPolicy(fqi)
        expected_reward = J(domain, fqi_policy, DISCOUNT_FACTOR, 50, N_j, seed=trajectory_size)
        print("J for FQI_{}: {}".format(trajectory_size, expected_reward))
        expected_FQI.append(expected_reward)

        # Parametric QLearning
        dataset = StateActionDataset(trajectories)
        loader = data.DataLoader(
            dataset,
            batch_size=None,
            sampler=data.BatchSampler(data.RandomSampler(dataset), 32, False)
        )

        pql = ParametricQLearning(get_net())
        pql.fit(loader, N_nn * 5) 

        pql_policy = ParameterQPolicy(pql)
        expected_reward = J(domain, pql_policy, DISCOUNT_FACTOR, 50, N_j, seed=trajectory_size)
        print("\nJ for PQL_{}: {}".format(trajectory_size, expected_reward))
        expected_PQ.append(expected_reward)


        # online
        online = OnlineParametricQLearning(get_net(), domain)
        online.fit(trajectory_size)

        pql_online = ParameterQPolicy(online)
        expected_reward = J(domain, pql_online, DISCOUNT_FACTOR, 50, N_j, seed=trajectory_size)
        print("\nJ for PQL-online_{}: {}".format(trajectory_size, expected_reward))
        expected_online.append(expected_reward)

    curve_plot(expected_FQI, expected_PQ, expected_online, trajectory_sizes)

if __name__ == '__main__':

    for i in range(10): # average across different seeds
        epsilon = 1e-3
        N = math.ceil(math.log((epsilon / B_r) * (1. - DISCOUNT_FACTOR), DISCOUNT_FACTOR))
        
        # Generate dataset
        trajectories = get_trajectories(80000, seed=i) # random trajectories
        dataset = StateActionDataset(trajectories)
        loader = data.DataLoader(
            dataset,
            batch_size=None,
            sampler=data.BatchSampler(data.RandomSampler(dataset), 32, False)
        )
        
        # Learn
        pql = ParametricQLearning(get_net())
        pql.fit(loader, 100) 

        # Make plot
        plot_Q(pql, "PQL_5")
        plot_mu(pql, "PQL_5")

        # Expected return
        policy = ParameterQPolicy(pql)
        domain = CarOnTheHillDomain(DISCOUNT_FACTOR, M, GRAVITY, TIME_STEP, INTEGRATION_TIME_STEP)
        j = J(domain, policy, DISCOUNT_FACTOR, 50, N, seed=i)
        print("\nReward PQL: ", j)

    print("\n----Protocol----\n")
    
    # Protocol
    protocol_comparison()