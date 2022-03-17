import matplotlib.pyplot as plt

def curve_plot(expected_FQI, expected_PQ, expected_online, trajectory_sizes):
    print("FQI")
    print(expected_FQI)
    print("PQ")
    print(expected_PQ)
    print("PQ-Online")
    print(expected_online)
    plt.plot(trajectory_sizes, expected_FQI, label="FQI")
    plt.plot(trajectory_sizes, expected_PQ, label="PQ")
    plt.plot(trajectory_sizes, expected_online,label="PQ-Online")
    plt.legend()
    plt.title("Expected return of the algorithm with respect to the trajectory size")
    plt.ylabel("Expected return J")
    plt.xlabel("Trajectory size")
    plt.savefig("figures/comparison.png")


curve_plot([-0.62, 0, 0, 0, 0],[0, 0, 0, 0, 0],[-0.82, -0.82, 0, 0, 0], [5000, 10000, 20000, 50000, 100000])