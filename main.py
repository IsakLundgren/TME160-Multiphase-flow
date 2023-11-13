import numpy as np
import matplotlib.pyplot as plt

# ---Task A---

# Particle properties
d_p = 0.5 * 1e-3  # m
rho_p = 2560  # kg/m^3
m_p = rho_p * 4 / 3 * np.pi * (d_p / 2) ** 3
Re_p_max = 0.1  # The maximum reynolds particle number where stokes flow occurs

# Fluid properties
mu_f = 1.0016 * 1e-3  # Pa s
rho_f = 1000  # kg/m^3

# Solver settings
t_0 = 0  # s
t_max = 1  # s
dt = 0.01  # s

U_f = 0  # m/s
V_0 = 0  # m/s
g = 9.82  # m/s^2

# Initialize quantities
N_steps = int((t_max - t_0) / dt)
t = t_0 * np.ones(N_steps)
V = V_0 * np.ones(N_steps)
Re_p = rho_f * np.abs(U_f - V_0) * d_p / mu_f * np.ones(N_steps)

# Run solver
for i in range(1, N_steps):
    # Explicit solver, calculating from previous timestep

    # Caluclate forces on the particle
    F_D = 3 * np.pi * mu_f * d_p * (U_f - V[i-1])  # N Drag force
    F_g = -m_p * g  # N Gravitational force
    # No lift force since there's no horizontal movement
    F_tot = F_D + F_g

    # Since there's no fluid movement, calculate added mass as a simple addition to particle mass
    m_added = 1 / 2 * m_p * rho_f / rho_p
    m_tot = m_p + m_added

    # Do timestep
    t[i] = t[i-1] + dt
    V[i] = V[i-1] + dt * F_tot / m_tot

    # Calculate reynolds number to ensure stokes flow
    Re_p[i] = rho_f * np.abs(U_f - V[i]) * d_p / mu_f

# Display and save results
_, ax = plt.subplots()
ax.plot(t, V, '-b', label='Y Velocity')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Velocity [m/s]')
ax.set_title(f'Velocity distribution in time, dt = {dt:.3g} s.')
ax.grid()
ax.legend()

_, ax = plt.subplots()
ax.plot(t, Re_p, '-b', label='Re_p')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Re_p [-]')
ax.set_title(f'Particle reynolds number distribution in time, dt = {dt:.3g} s.')
ax.grid()
ax.legend()

plt.show()
