from python.components import State
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14

flags = {
    'bbox_inches': 'tight'
}

def get_max_diff_r(domain, mdp):
    """Infinity norm of r"""
    n, m = domain.g.shape
    max_val = -1
    for x in range(n):
        for y in range(m):
            for action in domain.actions:
                diff = abs(mdp.r(State(x, y), action) - domain.r(State(x, y), action))
                if diff > max_val:
                    max_val = diff

    return max_val

def get_max_diff_p(domain, mdp):
    """Infinity norm of p"""
    n, m = domain.g.shape
    max_val = -1
    for x in range(n):
        for y in range(m):
            for action in domain.actions:
                for x_given in range(n):
                    for y_given in range(m):
                        diff = abs(mdp.p(State(x, y), State(x_given, y_given), action) - domain.p(State(x, y), State(x_given, y_given), action))
                        if diff > max_val:
                            max_val = diff

    return max_val

def get_max_diff_q(Q_hat, Q):
    """Infinity norm of q"""
    max_val = np.max(np.abs(Q_hat - Q))

    return max_val


def plot(x, y, estimate:str, file_name:str):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(x, y, '--bo')
    plt.xscale('log')
    plt.xlabel('$t$')
    plt.ylabel(f'$\\left\\| \\hat{{}} - {{}}_N \\right\\|_\\infty$'.format(estimate, estimate))
    plt.grid()
    plt.savefig("figures/{}".format(file_name), **flags)

    plt.close()