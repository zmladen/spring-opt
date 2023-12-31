from scipy.integrate import odeint
import numpy as np

# Placeholder values for thermal resistances and capacitances (for illustrative purposes)
R_C_SY = R_SY_SW = R_SW_ST = R_ST_PM = R_PM_A = R_SY_ST = R_SW_PM = 1  # Replace with actual thermal resistances
C_SY = C_SW = C_ST = C_PM = 1  # Replace with actual thermal capacitances


# Continuous-time state-space equations
def state_space_equations(x, t, A, B, u):
    """
    Define the state-space equations.
    :param x: State vector.
    :param t: Time variable.
    :param A: System matrix.
    :param B: Input matrix.
    :param u: Input vector (assumed constant over time for this example).
    :return: Derivative of the state vector.
    """
    dxdt = A.dot(x) + B.dot(u)
    return dxdt

# Define the system matrices A and B using the same placeholder values
A = np.array([
    [-(1/R_SY_SW + 1/R_SY_ST + 1/R_C_SY)/C_SY, 1/(R_SY_SW*C_SY), 1/(R_SY_ST*C_SY), 1/(R_C_SY*C_SY)],
    [1/(R_SY_SW*C_SW), -(1/R_SY_SW + 1/R_SW_ST + 1/R_SW_PM)/C_SW, 1/(R_SW_ST*C_SW), 1/(R_SW_PM*C_SW)],
    [1/(R_SY_ST*C_ST), 1/(R_SW_ST*C_ST), -(1/R_SY_ST + 1/R_SW_ST + 1/R_ST_PM)/C_ST, 1/(R_ST_PM*C_ST)],
    [1/(R_C_SY*C_PM), 1/(R_SW_PM*C_PM), 1/(R_ST_PM*C_PM), -(1/R_C_SY + 1/R_SW_PM + 1/R_ST_PM)/C_PM]
])
B = np.array([
    [1/C_SY, 0, 0, 0],
    [0, 1/C_SW, 0, 0],
    [0, 0, 1/C_ST, 0],
    [0, 0, 0, 1/C_PM]
])

# Define initial state and input
x0 = np.array([20, 20, 20, 20])  # Initial temperatures
u = np.array([10, 5, 15, 10])   # Power inputs

# Time points at which the solution should be computed
t = np.linspace(0, 60, 61)  # 60 seconds, sampled every second

# Solve the system of differential equations
x_t = odeint(state_space_equations, x0, t, args=(A, B, u))

# x_t contains the temperatures at each node at each time step
x_t[:5]  # Display the first five time steps as an example
