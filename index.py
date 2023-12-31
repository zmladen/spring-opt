import numpy as np

# Placeholder values for thermal resistances and capacitances (for illustrative purposes)
R_C_SY = R_SY_SW = R_SW_ST = R_ST_PM = R_PM_A = R_SY_ST = R_SW_PM = 1  # Replace with actual thermal resistances
C_SY = C_SW = C_ST = C_PM = 1  # Replace with actual thermal capacitances

# Continuous-time A matrix
A = np.array([
    [-(1/R_SY_SW + 1/R_SY_ST + 1/R_C_SY)/C_SY, 1/(R_SY_SW*C_SY), 1/(R_SY_ST*C_SY), 1/(R_C_SY*C_SY)],
    [1/(R_SY_SW*C_SW), -(1/R_SY_SW + 1/R_SW_ST + 1/R_SW_PM)/C_SW, 1/(R_SW_ST*C_SW), 1/(R_SW_PM*C_SW)],
    [1/(R_SY_ST*C_ST), 1/(R_SW_ST*C_ST), -(1/R_SY_ST + 1/R_SW_ST + 1/R_ST_PM)/C_ST, 1/(R_ST_PM*C_ST)],
    [1/(R_C_SY*C_PM), 1/(R_SW_PM*C_PM), 1/(R_ST_PM*C_PM), -(1/R_C_SY + 1/R_SW_PM + 1/R_ST_PM)/C_PM]
])

# Continuous-time B matrix
B = np.array([
    [1/C_SY, 0, 0, 0],
    [0, 1/C_SW, 0, 0],
    [0, 0, 1/C_ST, 0],
    [0, 0, 0, 1/C_PM]
])

# Sampling time Ts (in seconds)
Ts = 1

# Euler discretization of the A and B matrices
Phi = np.eye(len(A)) + Ts * A  # Approximate transition matrix using Euler method
H = Ts * B  # Approximate input matrix using Euler method

# Define the initial state (temperatures at each node) and input (power inputs)
x = np.array([20, 20, 20, 20])  # Initial temperatures (in degrees Celsius)
u = np.array([10, 5, 15, 10])  # Power inputs (in watts)

# Define the time horizon for simulation (e.g., 60 seconds)
time_horizon = 60  # Total time steps to simulate

# Initialize an array to store the state history
state_history = np.zeros((len(x), time_horizon + 1))
state_history[:, 0] = x  # Set the initial state

# Time-stepping loop for Euler discretization
for k in range(time_horizon):
    x = Phi.dot(x) + H.dot(u)  # State update equation
    state_history[:, k + 1] = x  # Store the state

# State history now contains the temperature at each node at each time step
state_history[:, :5]  # Display the first five time steps as an example
