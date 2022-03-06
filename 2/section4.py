from python.constants import *
from python.domain import ACTIONS, State, CarOnTheHillDomain
from python.policy import AlwaysAcceleratePolicy, RandomActionPolicy, FittedQPolicy
from python.simulation import Simulation
from python.fitted_Q import Fitted_Q

import numpy as np

from python.constants import *
from python.domain import CarOnTheHillDomain
from python.expected_return import J

import math
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from section3 import make_video

def get_stopping_rules():
    epsilon = 1e-3
    N = math.ceil(math.log((epsilon / (2 * B_r)) * (1. - DISCOUNT_FACTOR), DISCOUNT_FACTOR))

    return N, None

def get_models():
    LR = LinearRegression(n_jobs=-1)
    ETR = ExtraTreesRegressor(n_estimators=30, random_state=42)

    # TODO: MLP

    return ETR, LR

def get_trajectories(trajectory_length, N):
    # Random
    random_policy = RandomActionPolicy()
    random_trajectories = np.array([])

    random_trajectories = []

    for n in range(N):
        initial_state = State.random_initial_state(seed=n)
        simulation = Simulation(domain, random_policy, initial_state, remember_trajectory=True, seed=n, stop_when_terminal=True)
        simulation.simulate(trajectory_length)
        random_trajectories.extend(np.array(simulation.get_trajectory(values=True)).squeeze())
        print("\r ", n, "/", N, end="\r")

    random_trajectories = np.array(random_trajectories) 
    

    # TODO:

    return random_trajectories, None

def plot_Q(model, title:str):
    p = np.linspace(-1, 1, 200)  
    s = np.linspace(-3, 3, 600)

    for action in ACTIONS:
        pp, ss, uu = np.meshgrid(p[:-1], s[:-1], np.array([action]), indexing='ij')
        stateaction = np.vstack((pp.ravel(), ss.ravel(), uu.ravel())).T

        gridshape = pp.shape

        y_pred = model.predict(stateaction).reshape(gridshape[:-1])
        print(y_pred)

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

    for trajectories in get_trajectories(1000, 1000):
        for stopping_rule in get_stopping_rules():
            for model in get_models():
                fitted_Q = Fitted_Q(model, stopping_rule, DISCOUNT_FACTOR)
                fitted_Q.fit(trajectories)
                plot_Q(fitted_Q, model.__class__.__name__) # add stopping rule 
                plot_mu(fitted_Q, model.__class__.__name__)

                fitted_Q_policy = FittedQPolicy(fitted_Q)
                
                nb_simulation = 50

                j = J(domain, fitted_Q_policy, DISCOUNT_FACTOR, nb_simulation, N)
                print(j)
                exit()





