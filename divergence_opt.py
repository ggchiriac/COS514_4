import numpy as np
import math
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping, minimize
from scipy.integrate import quad

INT_LOWER_LIMIT = -20
INT_UPPER_LIMIT = 20
INT_SIZE = 10000
MIN_VAR = 0.001
MAX_VAR = 30
MIN_PDF = 1e-300
GRID = [-15, 15, 1000]


def p(x):
    pdf_1 = 0.4 * (1 / np.sqrt(2 * np.pi * 1)) * np.exp(-(x - 0) ** 2 / (2 * 1))
    pdf_2 = 0.3 * (1 / np.sqrt(2 * np.pi * 0.25)) * np.exp(-(x + 4) ** 2 / (2 * 0.25))
    pdf_3 = 0.3 * (1 / np.sqrt(2 * np.pi * 4)) * np.exp(-(x - 8) ** 2 / (2 * 4))

    return pdf_1 + pdf_2 + pdf_3


def q(x, mean, var):
    var = max(var, 0.001)
    pdf = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))

    return pdf


def forward_kl(z):
    x = np.random.uniform(INT_LOWER_LIMIT, INT_UPPER_LIMIT, INT_SIZE)

    # Clip array values for numerical stability
    p_clip = np.maximum(p(x), MIN_PDF)
    q_clip = np.maximum(q(x, z[0], z[1]), MIN_PDF)
    log_clip = np.log(p_clip / q_clip)
    prod = p(x) * log_clip

    return np.sum(prod) / INT_SIZE


def reverse_kl(z):
    x = np.random.uniform(INT_LOWER_LIMIT, INT_UPPER_LIMIT, INT_SIZE)

    # Clip array values for numerical stability
    p_clip = np.maximum(p(x), MIN_PDF)
    q_clip = np.maximum(q(x, z[0], z[1]), MIN_PDF)
    log_clip = np.log(q_clip / p_clip)
    prod = q(x, z[0], z[1]) * log_clip

    # Artificially make logarithm blow up when p goes to zero
    prod_adj = np.where(p_clip > MIN_PDF, prod, np.ones(prod.size) * np.inf)

    return np.sum(prod_adj) / INT_SIZE

def plot_q_star():
    z_0 = [0, 1]

    # Impose constraints for stability
    con = [{'type': 'ineq', 'fun': lambda z: z[1] - MIN_VAR},
           {'type': 'ineq', 'fun': lambda z: -z[1] + MAX_VAR},
           {'type': 'ineq', 'fun': lambda z: z[0] - INT_LOWER_LIMIT},
           {'type': 'ineq', 'fun': lambda z: -z[0] + INT_UPPER_LIMIT}]
    minimizer_kwargs = {"method": "SLSQP", "constraints": con}

    # Minimize forward KL
    res = basinhopping(forward_kl, z_0, minimizer_kwargs=minimizer_kwargs, niter=500)

    z_opt = res.x
    forward_kl_opt = res.fun

    print("Optimal mean and variance:", z_opt)
    print("Minimum value of forward KL:", forward_kl_opt)

    # Generate values over which to evaluate the PDF
    x = np.linspace(GRID[0], GRID[1], GRID[2])

    # Compute PDF values
    p_values = p(x)
    q_values = q(x, z_opt[0], z_opt[1])

    # Plot P and Q hat
    plt.plot(x, p_values, color='r', label = 'P')
    plt.plot(x, q_values, label = 'Q star')
    plt.title("Q star")
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.legend()
    plt.savefig("Qstar.png")
    plt.show()

def plot_q_hat():
    z_0 = [0, 1]

    # Impose constraints for stability
    con = [{'type': 'ineq', 'fun': lambda z: z[1] - MIN_VAR},
           {'type': 'ineq', 'fun': lambda z: -z[1] + MAX_VAR},
           {'type': 'ineq', 'fun': lambda z: z[0] - INT_LOWER_LIMIT},
           {'type': 'ineq', 'fun': lambda z: -z[0] + INT_UPPER_LIMIT}]
    minimizer_kwargs = {"method": "SLSQP", "constraints": con}

    # Minimize reverse KL
    res = basinhopping(reverse_kl, z_0, minimizer_kwargs=minimizer_kwargs, niter=500)

    z_opt = res.x
    reverse_kl_opt = res.fun

    print("Optimal mean and variance:", z_opt)
    print("Minimum value of reverse KL:", reverse_kl_opt)

    # Generate values over which to evaluate the PDF
    x = np.linspace(GRID[0], GRID[1], GRID[2])

    # Compute PDF values
    p_values = p(x)
    q_values = q(x, z_opt[0], z_opt[1])

    # Plot P and Q hat
    plt.plot(x, p_values, color='r', label = 'P')
    plt.plot(x, q_values, label = 'Q hat')
    plt.title("Q hat")
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.legend()
    plt.savefig("Qhat.png")
    plt.show()

def main():
    plot_q_star()
    plot_q_hat()

if __name__ == '__main__':
    main()
