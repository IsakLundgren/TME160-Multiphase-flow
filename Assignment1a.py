import numpy as np
import matplotlib.pyplot as plt

# ---Task A---

# Particle properties
d_p = 0.5 * 1e-3  # m
rho_p = 2560  # kg/m^3
m_p = rho_p * 4 / 3 * np.pi * (d_p / 2) ** 3
Re_p_max = 1e-1  # The maximum reynolds particle number where stokes flow occurs

# Fluid properties
mu_f = 1.0016 * 1e-3  # Pa s
rho_f = 1000  # kg/m^3

# Solver settings
t_0 = 0  # s
t_max = 0.4  # s
delta_t = 0.001  # s
delta_t_repeat = 0.01  # s

U_f = 0  # m/s
V_0 = 0  # m/s
g = 9.82  # m/s^2


# Solver quantity function
def getQuantities(dt, time, v, N_steps):
    reynolds = rho_f * np.abs(U_f - v) * d_p / mu_f

    if reynolds == 0:
        dragCoeff = 0
    elif reynolds < Re_p_max:
        dragCoeff = 24 / reynolds
    else:
        dragCoeff = 24 / reynolds * (1 + 0.15 * reynolds ** 0.687)

    drag = 1 / 2 * rho_f * d_p ** 2 * np.pi / 4 * dragCoeff * np.abs(U_f - v) * (U_f - v)  # N
    gravity = -m_p * g  # N
    pressureGrad = m_p * rho_f / rho_p * g  # N
    if time != 0:
        history = -m_p * (v - V_0) / np.sqrt(0.5 * time)
    else:
        history = 0

    return reynolds, dragCoeff, drag, gravity, pressureGrad, history


# Solve function
def solve(dt, schemeOrder=1):
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

    # Since there's no fluid movement, calculate added mass as a simple addition to particle mass
    m_added = 1 / 2 * m_p * rho_f / rho_p
    m_tot = m_p + m_added

    # Explicit solver, calculating from previous timestep
    for i in range(1, N_steps):
        # Caluclate forces on the particle
        Re_p[i-1], C_D[i-1], F_D[i-1], F_g[i-1], F_P[i-1], F_H[i-1] = getQuantities(dt, t[i-1], V[i-1], N_steps)
        F_tot[i-1] = F_D[i-1] + F_g[i-1] + F_P[i-1] + F_H[i-1]

        # Do timestep
        if schemeOrder == 1:
            # Euler's method
            t[i] = t[i-1] + dt
            V[i] = V[i-1] + dt * F_tot[i-1] / m_tot
        elif schemeOrder == 2:
            # Improved Euler's method
            t[i] = t[i-1] + dt

            V_tld = V[i-1] + dt * F_tot[i-1] / m_tot
            Re_p_tld, C_D_tld, F_D_tld, F_g_tld, F_P_tld, F_H_tld = getQuantities(dt, t[i-1], V_tld, N_steps)
            F_tot_tld = F_D_tld + F_g_tld + F_P_tld + F_H_tld

            # Do real timestep
            V[i] = V[i-1] + dt / 2 * (F_tot[i-1] + F_tot_tld) / m_tot

    # Post-process the added mass force
    F_A = np.zeros(N_steps)
    for i in range(1, N_steps):
        F_A[i-1] = 1 / 2 * m_p * rho_f / rho_p * (V[i-1] - V[i]) / dt

    # Assume equalized forces and copy over the last values
    Re_p[-1] = Re_p[-2]
    C_D[-1] = C_D[-2]
    F_D[-1] = F_D[-2]
    F_g[-1] = F_g[-2]
    F_P[-1] = F_P[-2]
    F_H[-1] = F_H[-2]
    F_A[-1] = F_A[-2]

    return t, V, Re_p, C_D, F_D, F_g, F_P, F_H, F_A


# Run solver
t_e1, V_e1, Re_p_e1, C_D_e1, F_D_e1, F_g_e1, F_P_e1, F_H_e1, F_A_e1 = solve(delta_t, 1)

# Fetch reference data
with open('ref/data_task1a.txt') as f:
    lines = f.readlines()
    t_ref = []
    v_ref = []
    for line in lines[1:]:
        lineEdited = line.replace('\n', '')
        lineSplit = lineEdited.split('\t')
        t_ref.append(float(lineSplit[0]))
        v_ref.append(-float(lineSplit[1]))

# Display and save results
figureDPI = 200
fig, ax = plt.subplots()
ax.plot(t_e1 * 1e3, V_e1, '-b', label='Y velocity')
ax.plot(t_ref, v_ref, '--r', label='Reference velocity')
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Velocity [m/s]')
ax.set_title(f'Velocity distribution in time, dt = {delta_t * 1e3:.3g} ms.')
ax.grid()
ax.legend()
fig.set_size_inches(8, 6)
fig.savefig('img/VelocityDistribution.png', dpi=figureDPI)

fig, ax = plt.subplots()
ax.plot(t_e1 * 1e3, Re_p_e1, '-b', label='Re_p')
# ax.hlines(Re_p_max, t[0], t[-1], colors='r', linestyles='--', label='Stokes regime')
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Re_p [-]')
ax.set_title(f'Particle reynolds number distribution in time, dt = {delta_t * 1e3:.3g} ms.')
ax.grid()
ax.legend()
fig.set_size_inches(8, 6)
fig.savefig('img/ReynoldsNo.png', dpi=figureDPI)

fig, ax = plt.subplots()
ax.plot(t_e1 * 1e3, F_g_e1 / np.abs(F_g_e1), '-b', label='Gravity cont.')
ax.plot(t_e1 * 1e3, F_D_e1 / np.abs(F_g_e1), '-r', label='Drag cont.')
ax.plot(t_e1 * 1e3, F_A_e1 / np.abs(F_g_e1), '-m', label='Added mass cont.')
ax.plot(t_e1 * 1e3, F_P_e1 / np.abs(F_g_e1), '-g', label='Pressure gradient cont.')
ax.plot(t_e1 * 1e3, F_H_e1 / np.abs(F_g_e1), '-y', label='History cont.')
ax.set_xlabel('Time [ms]')
ax.set_ylabel('F_x / |F_g| [-]')
ax.set_title(f'Normalized force distribution in time, dt = {delta_t * 1e3:.3g} ms.')
ax.grid()
ax.legend(loc='lower left')
fig.set_size_inches(8, 6)
fig.savefig('img/NormForce.png', dpi=figureDPI)

# Run new scheme
t_e1, V_e1, Re_p_e1, C_D_e1, F_D_e1, F_g_e1, F_P_e1, F_H_e1, F_A_e1 = solve(delta_t_repeat, 1)
t_e2, V_e2, Re_p_e2, C_D_e2, F_D_e2, F_g_e2, F_P_e2, F_H_e2, F_A_e2 = solve(delta_t_repeat, 2)

fig, ax = plt.subplots()
ax.plot(t_e1 * 1e3, V_e1, '-b', label='Euler\'s method')
ax.plot(t_e2 * 1e3, V_e2, '-g', label='Improved Euler\'s method')
ax.plot(t_ref, v_ref, '--r', label='Reference velocity')
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Velocity [m/s]')
ax.set_title(f'Velocity distribution in time, dt = {delta_t_repeat * 1e3:.3g} ms.')
ax.set_xlim(0, 100)
ax.grid()
ax.legend()
fig.set_size_inches(8, 6)
fig.savefig('img/SchemeComparison.png', dpi=figureDPI)

plt.show()
