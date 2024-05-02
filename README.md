# 2D Ising Model Monte Carlo Simulation

This project implements a Monte Carlo simulation of the 2D Ising model using the Metropolis algorithm. The Ising model is a mathematical model used to study the behavior of ferromagnetic materials and phase transitions.

## Overview

The 2D Ising model consists of a lattice of spins that can take values of +1 or -1. Each spin interacts with its nearest neighbors, and the system evolves according to the Metropolis algorithm. The simulation computes various physical quantities such as energy, magnetization, and specific heat as a function of temperature.

The Hamiltonian $\mathcal{H}$ for the 2D Ising model in the presence of an external magnetic field  $B$ is given by the following expression:

```math
\mathcal{H} = -J \sum_{\langle i, j \rangle} s_i s_j - \mu B \sum_i s_i
```

Here, $s_i$ represents the spin at site $i$, which can take values of `+1` or `-1`. $J$ is the interaction strength between nearest neighbor spins $\langle i, j \rangle$,  $\mu$ is the magnetic moment of each spin, and $B$ is the strength of the external magnetic field. The first sum runs over all nearest neighbor pairs of spins, while the second sum runs over all spins in the lattice.

## Features

- Implements the Metropolis algorithm for efficient Monte Carlo sampling
- Utilizes Numba JIT compilation for accelerated performance
- Computes energy, magnetization, and specific heat of the system
- Allows customization of lattice size, number of Monte Carlo sweeps, temperature range, and external magnetic field strength
- Provides visualizations of the computed physical quantities

## Dependencies

- Python 3.x
- NumPy
- Matplotlib
- Numba
- tqdm

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/MohamedElashri/IsingModel.git
   ```

2. Install the required dependencies:
   ```
   pip install numpy matplotlib numba tqdm
   ```

3. Run the simulation script:
   ```
   python src/IsingModel.py
   ```

4. Adjust the simulation parameters in the script as desired:
   - `L`: Lattice size (width)
   - `n`: Number of Monte Carlo sweeps
   - `Temperature`: Temperature range (includes critical temperature)
   - `B`: Strength of the external magnetic field
   - `mu`: Magnetic moment of each spin

## Jupyter Notebook

An interactive Jupyter Notebook version of the simulation is also available. You can run the notebook on Google Colab by clicking on the following badge:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohamedElashri/IsingModel/blob/main/Ising.ipynb)

## Results

The simulation generates plots of the physical quantities (energy, magnetization, and specific heat) as a function of temperature. The plots provide insights into the phase transition and critical behavior of the 2D Ising model.

## Optimization

Significant optimization efforts have been made to improve the performance of the simulation. The use of Numba's JIT compiler has greatly reduced the execution time, allowing for larger lattice sizes and more Monte Carlo sweeps. The code has been optimized to minimize nested loops and leverage vectorized operations.

## Installation on Apple Silicon (Mac M1)

To install Numba on a Mac M1 machine, follow these steps:

1. Ensure you are using the Python 3 version from Homebrew and not the one that comes with the OS. Add the following line to your `.bashrc` or `.zshrc` file:
   ```
   export PATH="/usr/local/opt/python/libexec/bin:$PATH"
   ```

2. Install the required packages:
   ```
   python3 -m pip install conda
   python3 -m pip install cytoolz
   python3 -m conda config --add channels conda-forge
   python3 -m conda install -c numba numba
   ```

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The Ising model implementation is based on the work of Wilhelm Lenz and Ernst Ising.
- The Metropolis algorithm is a widely used Monte Carlo method for simulating physical systems.

Feel free to customize and expand upon this README file to include any additional information or sections relevant to your project.
