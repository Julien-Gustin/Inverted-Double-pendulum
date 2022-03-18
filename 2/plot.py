import matplotlib.pyplot as plt
import matplotlib

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

def curve_plot2(expected_PQ_5, expected_PQ_200, trajectory_sizes):
    plt.plot(trajectory_sizes, expected_PQ_5, label="PQ 5 neurons")
    plt.plot(trajectory_sizes, expected_PQ_200, label="PQ 200 neurons")
    plt.legend()
    plt.title("Expected return of the algorithm with respect to the trajectory size")
    plt.ylabel("Expected return J")
    plt.xlabel("Trajectory size")
    plt.savefig("figures/comparison_2.png")


# curve_plot([-0.62, 0, 0, 0, 0, 0.32],[0.0005, -0.05, 0.27, 0, 0, 0],[0, -0.81, 0, 0, 0, 0, ], [5000, 10000, 20000, 50000, 100000, 500000])
curve_plot2([0.0005, -0.05, 0.27, 0, 0, 0],[0.0, 0.06, 0.2, 0.1, 0.4, 0.3], [5000, 10000, 20000, 50000, 100000, 500000])