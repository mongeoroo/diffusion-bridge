import numpy as np

# Example values for beta schedule (cumulative noise schedule)
T = 1000  # Total number of timesteps
betas = np.linspace(0.0001, 0.02, T)  # Linear beta schedule for simplicity

# Calculate cumulative product of (1 - betas) to get \bar{\alpha}_t
alphas = 1 - betas
alpha_bars = np.cumprod(alphas)  # \bar{\alpha}_t for each t

# Function to get noise variance at time step t
def get_noise_variance(t):
    return 1 - alpha_bars[t]

# Example: Get noise variance at T=400
t = 400
noise_variance_t = get_noise_variance(t)
print(f"Noise variance at T={t}: {noise_variance_t}")