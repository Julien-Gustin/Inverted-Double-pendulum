from python.constants import *
from python.domain import CarOnTheHillDomain
from python.policy import AlwaysAcceleratePolicy
from python.expected_return import J

import math
import matplotlib.pyplot as plt


plt.rcParams['font.size'] = 14

flags = {
    'bbox_inches': 'tight'
}


if __name__ == "__main__":
    epsilon = 1e-3
    N = math.ceil(math.log((epsilon / B_r), DISCOUNT_FACTOR))

    domain = CarOnTheHillDomain(DISCOUNT_FACTOR, M, GRAVITY, TIME_STEP, INTEGRATION_TIME_STEP)
    policy = AlwaysAcceleratePolicy()

    nb_simulations = 50

    Jn = J(domain, policy, DISCOUNT_FACTOR, nb_simulations, N, save=True)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(Jn)
    plt.title("Expected return of a policy for the car on thehill problem \nfrom 0 to N")

    plt.xlabel("N")
    plt.ylabel(r'$J^{\mu}_{\infty}$')
    plt.grid()
    plt.savefig("figures/Jn")

    plt.close()