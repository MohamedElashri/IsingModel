# coding: utf-8
# ----------------------------------------------------------------------
# 2d Ising model Monte-Carlo Simulation
# Author: Mohamed Elashri
# Email: elashrmr@mail.uc.edu
# Algorithm
#  1- Prepare some initial configrations of N spins.
#  2- Flip spin of a lattice site chosen randomly
#  3- Calculate the change in energy due to that
#  4- If this change is negative, accept such move. If change is positive, accept it with probability exp^{-dE/kT}
#  5- repeat 2-4.
# 6- calculate Other parameters and plot them
# ----------------------------------------------------------------------

"""
Lattice is a periodical structure of points that align one by one. 2D lattice can be plotted as:

* * * * * * * *
* * * * * * * *
* * * * * * * *
* * * * * * * *
* * * * * * * *

The points in lattice are called lattice points, neareast lattice points of point ^
are those lattice points denoted by (*) shown in the graph below:

* * *(*)* * * *
* *(*)^(*)* * *
* * *(*)* * * *
* * * * * * * *

Each lattice point is denoted by a number i in the Harmitonian.

The expression for the Energy of the total system is (online latex formula)
https://melashri.net/url/a or
(H = - J \sum_{ i = 0 }^{ N-1 }\sum_{ j = 0 }^{ N-1 } (s_{i,j}s_{i,j+1}+s_{i,j}s_{i+1,j}) )

* * * * * * * *
* * * * * * * *
* * * * * * * * <-the i-th lattice point
* * * * * * * *
* * * * * * * *

Periodical strcture means that lattice point at(1,1) is the same as that at(1,9) if the lattice is 5 by 8.
more e.g.(1,1)<=>(6,1),(2,3)<=>(2,11). A 2D lattice can be any Nx by Ny. The location (x,y)
here is another denotion of lattice point that is fundementally same as i-th lattice point denotation above.s

* * * * * * * * 4
* * * * * * * * 3
* * * * * * * * 2
* * * * * * * * 1
1 2 3 4 5 6 7 8

"""

# ----------------------------------------------------------------------
#  Import needed python libraries
# ----------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from numba import jit
from tqdm.notebook import tqdm

def initialize_lattice(L):
    """Initialize the lattice with random spin sites (+1 or -1) for up or down spins."""
    return np.random.choice([1, -1], size=(L, L))

@jit(nopython=True, cache=True)
def calcE(s, L, B, mu):
    """Calculate the energy of the lattice."""
    E = 0
    for i in range(L):
        for j in range(L):
            E += -dE(s, i, j, L, B, mu, include_neighbor_interaction=False) / 2
    E -= mu * B * s.sum()  # Magnetic field contribution
    return E / L ** 2

@jit(nopython=True)
def calcM(s, L):
    """Calculate the magnetization of a given configuration."""
    m = np.abs(s.sum())
    return m / L**2

@jit(nopython=True, cache=True)
def dE(s, i, j, L, B, mu, include_neighbor_interaction=True):
    """Calculate the interaction energy between spins."""
    t = s[i - 1 if i > 0 else L - 1, j]
    b = s[i + 1 if i < L - 1 else 0, j]
    l = s[i, j - 1 if j > 0 else L - 1]
    r = s[i, j + 1 if j < L - 1 else 0]
    
    neighbor_interaction = 2 * s[i, j] * (t + b + l + r) if include_neighbor_interaction else 0
    magnetic_contribution = 2 * mu * B * s[i, j]
    
    return neighbor_interaction + magnetic_contribution

@jit(nopython=True)
def mc(s, Temp, n, L, B, mu):
    """Perform Monte-Carlo sweeps."""
    for _ in range(n):
        i = random.randrange(L)  # Choose random row
        j = random.randrange(L)  # Choose random column
        ediff = dE(s, i, j, L, B, mu)
        if ediff <= 0:  # If the change in energy is negative
            s[i, j] = -s[i, j]  # Accept move and flip spin
        elif random.random() < np.exp(-ediff / Temp):  # Accept with probability exp(-dU/kT)
            s[i, j] = -s[i, j]
    return s

@jit(nopython=True)
def physics(s, T, n, L, B, mu):
    """Compute physical quantities."""
    En = 0
    En_sq = 0
    Mg = 0
    Mg_sq = 0
    for _ in range(n):
        s = mc(s, T, 1, L, B, mu)
        E = calcE(s, L, B, mu)
        M = calcM(s, L)
        En += E
        Mg += M
        En_sq += E * E
        Mg_sq += M * M
    En_avg = En / n
    mag = Mg / n
    CV = (En_sq / n - (En / n)**2) / (T**2)
    return En_avg, mag, CV

def simulate_ising_model(L, n, Temperature, B, mu):
    """Simulate the Ising model and compute physical quantities."""
    s = initialize_lattice(L)
    
    mag = np.zeros(len(Temperature))
    En_avg = np.zeros(len(Temperature))
    CV = np.zeros(len(Temperature))
    
    start = time.time()
    
    for ind, T in enumerate(tqdm(Temperature)):
        s = mc(s, T, n, L, B, mu)
        En_avg[ind], mag[ind], CV[ind] = physics(s, T, n, L, B, mu)
    
    end = time.time()
    elapsed_time = (end - start) / 60
    print(f'It took {elapsed_time:.2f} minutes to execute the code')
    
    return En_avg, mag, CV

def plot_results(En_avg, mag, CV, L, n, B, Temperature):
    """Plot the simulation results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(Temperature, En_avg, marker='.', color='IndianRed')
    axes[0].set_xlabel("Temperature (T)", fontsize=16)
    axes[0].set_ylabel("Energy", fontsize=16)
    axes[0].set_title("Average Energy vs Temperature", fontsize=18)
    
    axes[1].plot(Temperature, abs(mag), marker='.', color='RoyalBlue')
    axes[1].set_xlabel("Temperature (T)", fontsize=16)
    axes[1].set_ylabel("Magnetization", fontsize=16)
    axes[1].set_title("Magnetization vs Temperature", fontsize=18)
    
    axes[2].plot(Temperature, CV, marker='.', color='IndianRed')
    axes[2].set_xlabel("Temperature (T)", fontsize=16)
    axes[2].set_ylabel("Specific Heat", fontsize=16)
    axes[2].set_title("Specific Heat vs Temperature", fontsize=18)
    
    plt.suptitle(f"Simulation of 2D Ising Model by Metropolis Algorithm\n"
                 f"Lattice Dimension: {L}x{L}, External Magnetic Field(B)={B}, Metropolis Step={n}",
                 fontsize=20)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
#                                Usage
# ----------------------------------------------------------------------


# Define parameters
L = 50  # Lattice size (width)
n = 1000 * L**2  # Number of MC sweeps
Temperature = np.arange(1.6, 3.25, 0.01)  # Temperature range (includes critical temperature)
B = 0.1  # Strength of the magnetic field
mu = 1  # Magnetic moment of each spin

# Run the simulation and plot the results
En_avg, mag, CV = simulate_ising_model(L, n, Temperature, B, mu)
plot_results(En_avg, mag, CV, L, n, B, Temperature)