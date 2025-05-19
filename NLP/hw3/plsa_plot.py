import matplotlib.pyplot as plt

# Load data
iterations = []
likelihoods = []
with open("likelihood_log.log") as f:
    next(f)  # Skip header
    for line in f:
        iter_num, ll = line.strip().split("\t")
        iterations.append(int(iter_num))
        likelihoods.append(float(ll))

# Plot
plt.figure(figsize=(10, 5))
plt.plot(iterations, likelihoods, label="Log-Likelihood")
plt.xlabel("Iteration")
plt.ylabel("Log-Likelihood")
plt.title("PLSA Log-Likelihood Convergence")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plsa_log_likelihood.png")
plt.show()
