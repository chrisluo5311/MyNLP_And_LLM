import matplotlib.pyplot as plt

# Load the uploaded log file
likelihoods = []
with open("likelihood_log.log", "r") as f:
    next(f)  # Skip header
    for line in f:
        _, val = line.strip().split()
        likelihoods.append(float(val))

iterations = list(range(1, len(likelihoods) + 1))
delta = [0] + [j - i for i, j in zip(likelihoods[:-1], likelihoods[1:])]

# Plot delta chart
plt.figure(figsize=(14, 6))
plt.plot(iterations, delta, label="Δ Log-Likelihood", color='blue')
plt.axhline(0, linestyle='--', color='gray')
plt.xlabel("Iteration")
plt.ylabel("Change in Log-Likelihood")
plt.title("PLSA EM: Log-Likelihood Delta per Iteration (500 Iterations)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plsa_log_likelihood_delta.png")
plt.show()

# Plot summary table
fig, ax = plt.subplots(figsize=(6, 2))
table_data = [
    ["Start (1)", f"{likelihoods[0]:,.2f}", "N/A"],
    ["Midpoint (250)", f"{likelihoods[249]:,.2f}", f"{likelihoods[249] - likelihoods[248]:,.2f}"],
    ["End (500)", f"{likelihoods[-1]:,.2f}", f"{likelihoods[-1] - likelihoods[-2]:,.2f}"]
]
col_labels = ["Iteration", "Log-Likelihood", "Delta from Previous"]
ax.axis('off')
table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
table.scale(1, 2)
plt.title("PLSA Log-Likelihood Summary (500 Iterations)")
plt.tight_layout()
plt.savefig("plsa_log_likelihood_summary.png")
plt.show()
