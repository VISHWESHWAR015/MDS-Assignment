from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
X, y = make_moons(n_samples=300, noise=0.05, random_state=0)

# DBSCAN clustering
# eps: Maximum distance between two samples for them to be considered as in the same neighborhood
# min_samples: Minimum number of samples in a neighborhood to form a core point
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit_predict(X)

# Plot the results
plt.figure(figsize=(8, 6))

# Identify core, border, and noise points
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True

# Plot core points
plt.scatter(X[core_samples_mask, 0], X[core_samples_mask, 1], 
            c=labels[core_samples_mask], cmap='viridis', s=50, label='Core Points')

# Plot border/noise points
plt.scatter(X[~core_samples_mask, 0], X[~core_samples_mask, 1], 
            c=labels[~core_samples_mask], cmap='viridis', s=50, label='Border/Noise Points', alpha=0.5)

plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
