import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
from torchdiffeq import odeint
import matplotlib.pyplot as plt

# Yes, the training process remains largely unchanged, but the way you use the model
# for prediction shifts slightly. During training, the model learns the dynamics of the system
# by fitting to the entire trajectory of states (positions and velocities) over time. 
# In prediction, however, you input a specific initial state (z_init) and a target time
# to predict the state at that specific future time.


# Generate training data using your existing simulation
# ...
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
damped_solution = solve_ivp(damped_spring_mass_system, t_span, z0, args=(m_known, k_known, c_known), t_eval=t_eval, atol=1e-10, rtol=1e-10)

# Standard deviation for the noise
noise_std = 0.05  # This value can be adjusted based on the desired noise level

# Add Gaussian noise to the displacement data
noisy_displacement = damped_solution.y[0] + np.random.normal(0, noise_std, damped_solution.y[0].shape)

# Noisy sample data for parameter identification
noisy_sample_data = np.vstack((damped_solution.t, noisy_displacement))

print("noisy_sample_data shape", noisy_sample_data.shape)


# Define a neural network model for the system's dynamics
class NeuralODE(nn.Module):
    def __init__(self):
        super(NeuralODE, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )

    def forward(self, t, z):
        return self.net(z)

# Initialize the model
model = NeuralODE()

# Define an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Prepare the data (convert numpy arrays to torch tensors)
t_tensor = torch.tensor(damped_solution.t, dtype=torch.float32)
z_tensor = torch.tensor(damped_solution.y.T, dtype=torch.float32)  # Transpose to match (N, 2) shape

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    z_pred = odeint(model, z_tensor[0], t_tensor)
    loss = torch.mean((z_pred - z_tensor)**2)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Plot the results
z_pred_np = z_pred.detach().numpy()
plt.figure(figsize=(12, 8))
plt.plot(t_tensor.numpy(), z_tensor[:, 0].numpy(), label='Original Data', color='blue')
plt.plot(t_tensor.numpy(), z_pred_np[:, 0], label='Predicted Data', linestyle='--', color='orange')
plt.xlabel('Time (seconds)')
plt.ylabel('Displacement (meters)')
plt.title('Comparison of Original and Predicted Data')
plt.legend()
plt.grid(True)
plt.show()
