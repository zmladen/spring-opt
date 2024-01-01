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
noise_std = 0.0005  # This value can be adjusted based on the desired noise level

# Add Gaussian noise to the displacement and velocity data
noisy_displacement = damped_solution.y[0] + np.random.normal(0, noise_std, damped_solution.y[0].shape)
noisy_velocity = damped_solution.y[1] + np.random.normal(0, noise_std, damped_solution.y[1].shape)

# Noisy sample data for parameter identification
noisy_sample_data = np.vstack((damped_solution.t, noisy_displacement, noisy_velocity))

# Objective function for parameter identification
def safe_objective_for_identification(params, observed_data):
    m, k, c = params
    if m <= 0 or k <= 0 or c < 0:
        return np.inf

    t_observed = observed_data[0]  # Time points
    x_observed = observed_data[1]  # Displacement data
    v_observed = observed_data[2]  # Velocity data

    try:
        solution = solve_ivp(damped_spring_mass_system, [t_observed[0], t_observed[-1]], z0, args=(m, k, c), t_eval=t_observed)
        x_predicted = solution.y[0]
        v_predicted = solution.y[1]
    except Exception as e:
        return np.inf

    error_displacement = np.sum((x_observed - x_predicted)**2)
    error_velocity = np.sum((v_observed - v_predicted)**2)
    return error_displacement + error_velocity

# Initial guess and bounds for the optimization
initial_guess = [0.5, 0.5, 0.5]
bounds = Bounds([0.1, 0.1, 0], [10, 10, 5])

# Perform the optimization using noisy data
result_noisy = minimize(safe_objective_for_identification, initial_guess, args=(noisy_sample_data,), 
                        method='trust-constr', bounds=bounds)

# Extract the estimated parameters with noisy data
estimated_parameters_noisy = result_noisy.x

print("estimated_parameters_noisy", estimated_parameters_noisy)

# Plotting both the original data and the noisy data
plt.figure(figsize=(12, 8))
plt.plot(damped_solution.t, damped_solution.y[0], label='Original Displacement (Known Parameters)', color='blue')
plt.plot(damped_solution.t, damped_solution.y[1], label='Original Velocity (Known Parameters)', color='green')
plt.plot(noisy_sample_data[0], noisy_sample_data[1], label='Noisy Displacement', linestyle='--', color='orange')
plt.plot(noisy_sample_data[0], noisy_sample_data[2], label='Noisy Velocity', linestyle=':', color='purple')
plt.xlabel('Time (seconds)')
plt.ylabel('Displacement and Velocity')
plt.title('Original vs Noisy Data for the Damped Spring-Mass System')
plt.legend()
plt.grid(True)
plt.show()

# Output the estimated parameters
print("Estimated Parameters with Noisy Data: Mass = {:.2f}, Spring Constant = {:.2f}, Damping Coefficient = {:.2f}".format(*estimated_parameters_noisy))
