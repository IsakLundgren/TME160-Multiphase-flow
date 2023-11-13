import numpy as np
import matplotlib.pyplot as plt

# ---Task A---

# Particle properties
d_p = 0.5 * 1e-3  # m
rho_p = 2560  # kg/m^3
m_p = rho_p * 4 / 3 * np.pi * (d_p / 2) ** 3

# Fluid properties
mu_f = 1.0016 * 1e-3  # Pa s
rho_f = 1000  # kg/m^3

# Solver settings
t_0 = 0  # s
t_max = 2  # s
dt = 0.01  # s

U_f = 0  # m/s
V_0 = 0  # m/s
g = 9.82  # m/s^2

# Initialize quantities
t = t_0
V = [V_0]
Re_p = [rho_f * np.abs(U_f - V_0) * d_p / mu_f]

# Run solver
for i in range(0, int((t_max - t_0) / dt)):
    # Explicit solver, calculating from previous timestep

    # Caluclate forces on the particle
    F_D = 3 * np.pi * mu_f * d_p * (U_f - V[i])
