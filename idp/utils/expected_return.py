import numpy as np

def J(env, policy, discount_factor:float, nb_simulations: int, trajectory_length:int):
    """ Estimates the expected return of a policy given an environment """
    J_hat = np.zeros((nb_simulations))
    for i in range(nb_simulations):
        env.seed(i)
        state = env.reset()

        done = False
        j = 0
        n = 0

        while not done and n < trajectory_length:
            action = policy.compute_optimal_actions(state)
            state, reward, done, _ = env.step([action])
            j += (discount_factor ** n) * reward
            n += 1   

        J_hat[i] = j

    return J_hat.mean(), J_hat.std()