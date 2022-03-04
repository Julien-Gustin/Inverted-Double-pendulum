from python.constants import *
from python.domain import State, CarOnTheHillDomain

if __name__ == "__main__":
    domain = CarOnTheHillDomain(DISCOUNT_FACTOR, M, GRAVITY, TIME_STEP, INTEGRATION_TIME_STEP)
    initial_state = State.random_initial_state()
    print(initial_state)
