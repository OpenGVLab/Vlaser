"""
Generate Gamma-distributed samples without using scipy, proposed by pi0 paper

"""

import matplotlib.pyplot as plt
import torch

# Parameters
s = 0.999
alpha = 1.5
beta = 1.0
n_sample = int(1e5)

# use gamma: https://math.stackexchange.com/questions/190670/how-exactly-are-the-beta-and-gamma-distributions-related
gamma_alpha_dist = torch.distributions.Gamma(alpha, 1)
gamma_beta_dist = torch.distributions.Gamma(beta, 1)

x = gamma_alpha_dist.sample((n_sample,))
y = gamma_beta_dist.sample((n_sample,))
z = x / (x + y)
scaled_samples = s * (1 - z)

print("Min:", scaled_samples.min())
print("Max:", scaled_samples.max())

plt.figure()
plt.hist(scaled_samples, bins=30, density=True, alpha=0.7, color="blue")
plt.xlabel("τ")
plt.ylabel("Density")
plt.title(f"Samples from Beta((s-τ)/s; {alpha}, {beta}) with s={s}")
plt.grid(True)
plt.savefig("beta_samples_from_gamma.png")

# use beta directly
beta_dist = torch.distributions.Beta(alpha, beta)
samples = beta_dist.sample((n_sample,))
scaled_samples = s * (1 - samples)
print("Min:", scaled_samples.min())
print("Max:", scaled_samples.max())

plt.figure()
plt.hist(scaled_samples, bins=30, density=True, alpha=0.7, color="blue")
plt.xlabel("τ")
plt.ylabel("Density")
plt.title(f"Samples from Beta((s-τ)/s; {alpha}, {beta}) with s={s}")
plt.grid(True)
plt.savefig("beta_samples.png")
