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
t_max = 0.1  # s
dt = 0.001  # s

U_f = 0  # m/s
V_0 = 0  # m/s
g = 9.82  # m/s^2

# Initialize quantities
N_steps = int((t_max - t_0) / dt)
t = t_0 * np.ones(N_steps)
V = V_0 * np.ones(N_steps)
Re_p = rho_f * np.abs(U_f - V_0) * d_p / mu_f * np.ones(N_steps)
C_D = np.ones(N_steps)
F_D = np.zeros(N_steps)
F_g = np.zeros(N_steps)
F_P = np.zeros(N_steps)
F_H = np.zeros(N_steps)
F_tot = np.zeros(N_steps)

# Run solver
for i in range(1, N_steps):
    # Explicit solver, calculating from previous timestep

    # Calculate drag coefficient
    if Re_p[i-1] == 0:
        C_D[i-1] = 0
    elif Re_p[i-1] < Re_p_max:
        C_D[i-1] = 24 / Re_p[i-1]
    else:
        C_D[i-1] = 24 / Re_p[i-1] * (1 + 0.15 * Re_p[i-1] ** 0.687)

    # Caluclate forces on the particle
    F_D[i-1] = 1 / 2 * rho_f * d_p ** 2 * np.pi / 4 * C_D[i-1] * np.abs(U_f - V[i-1]) * (U_f - V[i-1])  # N Drag force
    F_g[i-1] = -m_p * g  # N Gravitational force
    F_P[i-1] = m_p * rho_f / rho_p * g  # N Pressure gradient force
    F_H[i-1] = m_p * (V[i-1] - V[0]) / np.sqrt(0.5 * (t_0 + dt * N_steps))
    F_tot[i-1] = F_D[i-1] + F_g[i-1] + F_P[i-1] + F_H[i-1]

    # Since there's no fluid movement, calculate added mass as a simple addition to particle mass
    m_added = 1 / 2 * m_p * rho_f / rho_p
    m_tot = m_p + m_added

    # Do timestep
    t[i] = t[i-1] + dt
    V[i] = V[i-1] + dt * F_tot[i-1] / m_tot

    # Calculate reynolds number to ensure stokes flow
    Re_p[i] = rho_f * np.abs(U_f - V[i]) * d_p / mu_f

# Post-process the added mass force
F_A = np.zeros(N_steps)
for i in range(1, N_steps):
    F_A[i-1] = 1 / 2 * m_p * rho_f / rho_p * (V[i-1] - V[i]) / dt

# Assume equalized forces and copy over the last values
F_D[-1] = F_D[-2]
F_g[-1] = F_g[-2]
F_P[-1] = F_P[-2]
F_H[-1] = F_H[-2]
F_A[-1] = F_A[-2]

# Display and save results
figureDPI = 200
fig, ax = plt.subplots()
ax.plot(t * 1e3, V, '-b', label='Y Velocity')
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Velocity [m/s]')
ax.set_title(f'Velocity distribution in time, dt = {dt:.3g} s.')
ax.grid()
ax.legend()
fig.set_size_inches(8, 6)
fig.savefig('img/VelocityDistribution.png', dpi=figureDPI)

fig, ax = plt.subplots()
ax.plot(t * 1e3, Re_p, '-b', label='Re_p')
# ax.hlines(Re_p_max, t[0], t[-1], colors='r', linestyles='--', label='Stokes regime')
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Re_p [-]')
ax.set_title(f'Particle reynolds number distribution in time, dt = {dt:.3g} s.')
ax.grid()
ax.legend()
fig.set_size_inches(8, 6)
fig.savefig('img/ReynoldsNo.png', dpi=figureDPI)

fig, ax = plt.subplots()
ax.plot(t * 1e3, F_g / np.abs(F_g), '-b', label='Gravity cont.')
ax.plot(t * 1e3, F_D / np.abs(F_g), '-r', label='Drag cont.')
ax.plot(t * 1e3, F_A / np.abs(F_g), '-m', label='Added mass cont.')
ax.plot(t * 1e3, F_P / np.abs(F_g), '-g', label='Pressure gradient cont.')
ax.plot(t * 1e3, F_H / np.abs(F_g), '-y', label='History cont.')
ax.set_xlabel('Time [ms]')
ax.set_ylabel('F_x / |F_g| [-]')
ax.set_title(f'Normalized force distribution in time, dt = {dt:.3g} s.')
ax.grid()
ax.legend(loc='lower left')
fig.set_size_inches(8, 6)
fig.savefig('img/NormForce.png', dpi=figureDPI)

plt.show()
