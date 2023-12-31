import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt

# Known parameters for the spring-mass system (for simulation)
m_known = 1.0  # Mass (kg)
k_known = 2.0  # Spring constant (N/m)
c_known = 0.5  # Damping coefficient (N·s/m)

# State-space representation of the damped spring-mass system
def damped_spring_mass_system(t, z, m, k, c):
    x, x_dot = z
    return [x_dot, -k/m * x - c/m * x_dot]

# Initial conditions (initial displacement and velocity)
z0 = [1.0, 0]

# Time span for the simulation
t_span = [0, 10]  # 10 seconds
t_eval = np.linspace(t_span[0], t_span[1], 100)

# Simulate the system
damped_solution = solve_ivp(damped_spring_mass_system, t_span, z0, args=(m_known, k_known, c_known), t_eval=t_eval)

# Standard deviation for the noise
noise_std = 0.05  # This value can be adjusted based on the desired noise level

# Add Gaussian noise to the displacement data
noisy_displacement = damped_solution.y[0] + np.random.normal(0, noise_std, damped_solution.y[0].shape)

# Noisy sample data for parameter identification
noisy_sample_data = np.vstack((damped_solution.t, noisy_displacement))

# Objective function for parameter identification with known mass
def objective_with_known_mass(params, observed_data, known_mass):
    k, c = params
    if k <= 0 or c < 0:
        return np.inf

    t_observed = observed_data[0]  # Time points
    x_observed = observed_data[1]  # Displacement data

    try:
        solution = solve_ivp(damped_spring_mass_system, [t_observed[0], t_observed[-1]], z0, args=(known_mass, k, c), t_eval=t_observed)
        x_predicted = solution.y[0]
    except Exception as e:
        return np.inf

    return np.sum((x_observed - x_predicted)**2)

# Known mass value
known_mass = 1.0  # Mass is known and fixed at 1 kg

# Initial guess for the remaining parameters (k and c)
initial_guess_k_c = [0.5, 0.5]  # These values can be adjusted

# Bounds for k and c
bounds_k_c = Bounds([0.1, 0], [10, 5])

# Perform the optimization using noisy data and the known mass
result_known_mass = minimize(objective_with_known_mass, initial_guess_k_c, args=(noisy_sample_data, known_mass), 
                             method='trust-constr', bounds=bounds_k_c)


# Extract the estimated parameters with known mass
estimated_k, estimated_c = result_known_mass.x

# Simulate the system with the estimated parameters for comparison
estimated_solution = solve_ivp(damped_spring_mass_system, t_span, z0, args=(known_mass, estimated_k, estimated_c), t_eval=t_eval)

# Plotting both the original data, the estimation, and the noisy data
plt.figure(figsize=(12, 8))
plt.plot(damped_solution.t, damped_solution.y[0], label='Original Data (Known Parameters)', color='blue')
plt.plot(estimated_solution.t, estimated_solution.y[0], label='Estimation (Estimated Parameters)', linestyle='--', color='red')
plt.plot(noisy_sample_data[0], noisy_sample_data[1], label='Noisy Data', linestyle=':', color='orange')
plt.xlabel('Time (seconds)')
plt.ylabel('Displacement (meters)')
plt.title('Comparison of Damped Spring-Mass System: Original Data vs Estimation vs Noisy Data')

# Include input parameters (known values) and estimated parameters in the legend
input_params_legend = f'Input Parameters (Known):\nMass = {m_known} kg\nSpring Constant = {k_known} N/m\nDamping Coefficient = {c_known} N·s/m'
estimated_params_legend = f'Estimated Parameters:\nSpring Constant = {estimated_k:.2f} N/m\nDamping Coefficient = {estimated_c:.2f} N·s/m'
plt.legend([input_params_legend, estimated_params_legend, 'Noisy Data'], loc='upper right', fontsize=10)

plt.grid(True)
plt.show()


# Output the estimated parameters
print("Estimated Parameters with Known Mass: Spring Constant = {:.2f}, Damping Coefficient = {:.2f}".format(estimated_k, estimated_c))