from python.constants import *
from python.domain import ACTIONS, State, CarOnTheHillDomain
from python.policy import AlwaysAcceleratePolicy, RandomActionPolicy, FittedQPolicy
from python.simulation import Simulation
from python.fitted_Q import Fitted_Q
from python.neural_network import NN
import numpy as np

from python.constants import *
from python.domain import CarOnTheHillDomain
from python.expected_return import J

import math
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor

def infinity_norm(Q_hat, Q):
    """Infinity norm of q"""
    max_val = np.max(np.abs(Q_hat - Q))
    return max_val

# remove yield
def bound_stop(current_n, prev_Q_hat, current_Q_hat):
    epsilon = 1e-2
    N = math.ceil(math.log((epsilon / (2 * B_r)) * (1. - DISCOUNT_FACTOR), DISCOUNT_FACTOR))
    return current_n > N

def distance_stop(current_n, prev_Q_hat, current_Q_hat):
    epsilon = 0.45

    if prev_Q_hat is None or current_Q_hat is None:
        return False

    print( infinity_norm(prev_Q_hat, current_Q_hat))

    return infinity_norm(prev_Q_hat, current_Q_hat) <= epsilon


def get_stopping_rules():
    return (distance_stop, "distance"), (bound_stop, "bound")

def get_models():
    LR =  lambda : LinearRegression(n_jobs=-1)
    ETR = lambda : ExtraTreesRegressor(n_estimators=30, random_state=42)
    NEURAL_NET = lambda : NN(layers=3, neurons=8, output=1, epochs=10, batch_size=32, activation="relu")

    return [NEURAL_NET]#, ETR, LR

def get_trajectories(nb_p, nb_s):
    # Random
    domain = CarOnTheHillDomain()
    random_policy = RandomActionPolicy()
    random_trajectories = np.array([])

    random_trajectories = []
    
    trajectory_length = 250
    n = 0
    buffer_size = nb_p * nb_s * 2
    while len(random_trajectories) <= buffer_size:
        initial_state = State.random_initial_state(seed=n)
        simulation = Simulation(domain, random_policy, initial_state, remember_trajectory=True, seed=n, stop_when_terminal=True)
        simulation.simulate(trajectory_length)
        random_trajectories.extend(np.array(simulation.get_trajectory(values=True)).squeeze())
        print("\r ", len(random_trajectories), "/", buffer_size, end="\r")
        n += 1

    random_trajectories = np.array(random_trajectories[:buffer_size]) 
    
    #the following will just contain a discretization of the state space in order to have an "exhaustive" list of length #X times #U
    p_discretized = np.linspace(-1, 1, nb_p)
    s_discretized = np.linspace(-3, 3, nb_s)
    actions = ACTIONS

    possible_trajectories = []

    for p in p_discretized:
        for s in s_discretized:
            for u in actions:
                starting_state = State(p, s)
                reached_state = domain.f(starting_state, u)
                reward = domain.r(starting_state, u)
                possible_trajectories.append(np.array([*starting_state.values(), u, reward, *reached_state.values()]))

    possible_trajectories = np.array(possible_trajectories)

    return (random_trajectories, "random one-steps"), (possible_trajectories, "exhaustive list")

def plot_Q(model, title:str):
    p = np.linspace(-1, 1, 100)  
    s = np.linspace(-3, 3, 300)

    for action in ACTIONS:
        pp, ss, uu = np.meshgrid(p[:-1], s[:-1], np.array([action]), indexing='ij')
        stateaction = np.vstack((pp.ravel(), ss.ravel(), uu.ravel())).T

        gridshape = pp.shape

        y_pred = model.predict(stateaction).reshape(gridshape[:-1])

        plt.pcolormesh(
        p[:-1], s[:-1], y_pred.T,
        cmap='coolwarm_r',
        vmin=-1, vmax=1,
        rasterized=True,
        shading="auto",)

        plt.colorbar()

        plt.xlabel(r'$p$')
        plt.ylabel(r'$s$')

        plt.savefig("figures/" + "Q" + title + str(action) + ".png")
        plt.close()

def plot_mu(model, title:str):
    p = np.linspace(-1, 1, 200)
    s = np.linspace(-3, 3, 600)

    pp, ss = np.meshgrid(p[:-1], s[:-1], indexing='ij')
    states = np.vstack((pp.ravel(), ss.ravel())).T

    gridshape = pp.shape

    y_pred = model.compute_optimal_actions(states).reshape(gridshape)

    plt.pcolormesh(
    p[:-1], s[:-1], y_pred.T,
    cmap='coolwarm_r',
    vmin=-4, vmax=4,
    rasterized=True,
    shading="auto",)

    plt.xlabel(r'$p$')
    plt.ylabel(r'$s$')

    plt.savefig("figures/" + "mu" + title + ".png")
    plt.close()


if __name__ == "__main__":
    domain = CarOnTheHillDomain(DISCOUNT_FACTOR, M, GRAVITY, TIME_STEP, INTEGRATION_TIME_STEP)
    epsilon = 1e-3
    N = math.ceil(math.log((epsilon / (2 * B_r)) * (1. - DISCOUNT_FACTOR), DISCOUNT_FACTOR))

    for trajectories in get_trajectories(115, 345):
        for stopping_rule in get_stopping_rules():
            for get_model in get_models():
                trajectory, trajectory_label = trajectories
                rule, rule_label = stopping_rule

                title = get_model().__class__.__name__ + "_" + trajectory_label + "_" + rule_label

                print(title)

                fitted_Q = Fitted_Q(get_model, rule, DISCOUNT_FACTOR)
                fitted_Q.fit(trajectory)
                plot_Q(fitted_Q, title)
                plot_mu(fitted_Q, title)

                fitted_Q_policy = FittedQPolicy(fitted_Q)
                
                nb_simulation = 50
                j = J(domain, fitted_Q_policy, DISCOUNT_FACTOR, nb_simulation, N)
                print("Expected return of {} using {} trajectory generator and {} stopping rule:\n\t {}".format(get_model().__class__.__name__, trajectory_label, rule_label, j))





