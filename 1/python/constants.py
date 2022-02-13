import numpy as np 

# Instance of the domain
G = np.array(
            [
                [-3, 1, -5, 0, 19],
                [6, 3, 8, 9, 10],
                [5, -8, 4, 1, -8],
                [6, -9, 4, 19, -5],
                [-20, -17, -4, -4, 9]
            ]
        )

# Bound on the reward
Br = G.max() 

# Decay factor
GAMMA = 0.99

# Probability distribution
W = [0.5, 0.5] #first element is the probability to make the right action, the second is the probability to go back to (0, 0)