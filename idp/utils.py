import numpy as np

def generate_sample(env, buffer_size, actions, seed):
    np.random.seed(seed)

    buffer = []
    prec_state = env.reset()

    while len(buffer) < buffer_size:
        action = env.action_space.sample() #np.random.choice(actions)
        state, reward, done, _ = env.step([action])

        buffer.append([prec_state, action, reward, state, done])
        if done:
            state = env.reset()

        prec_state = state
    return buffer[:buffer_size]

def get_discretize_action(num):
    if num % 2 == 0:
        print("error, should be even")
        exit()

    return np.linspace(-1, 1, num)


def J(env, policy, discount_factor:float, nb_simulations: int, trajectory_length:int):
    """ Estimates the expected return of a policy given an environment """
    J_hat = np.zeros((nb_simulations))

    for i in range(nb_simulations):
        state = env.reset()

        done = False
        j = 0
        n = 0
        while not done and n < trajectory_length :
            action = policy.compute_optimal_actions([state])
            state, reward, done, _ = env.step([action])
            j += (discount_factor ** n) * reward
            n += 1   

        J_hat[i] = j

    return J_hat.mean()