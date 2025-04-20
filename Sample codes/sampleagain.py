# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import time

# Load dataset
data = pd.read_csv("healthcare_data.csv")  
hospitals = data["Hospital_code"].values
severity_mapping = {'Extreme': 3, 'Moderate': 2, 'Minor': 1}
severity = data["Severity of Illness"].map(severity_mapping).values
max_capacity = data["Available Extra Rooms in Hospital"].values

# Normalize severity scores
severity_normalized = severity / np.sum(severity)

# Objective function
def objective_function(x):
    allocation_penalty = np.sum(np.maximum(0, x - max_capacity) ** 2)
    severity_penalty = np.sum((x / np.sum(x) - severity_normalized) ** 2)
    severity_weight = 15
    return severity_weight * severity_penalty + allocation_penalty

# Firefly Algorithm
class FireflyAlgorithm:
    def __init__(self, n_fireflies, n_dim, lower_bound, upper_bound, max_iter, alpha=0.5, beta_min=0.2, gamma_val=1.0):
        self.n_fireflies = n_fireflies
        self.n_dim = n_dim
        self.lower_bound = lower_bound
        self.upper_bound = np.where(upper_bound == 0, 1e-6, upper_bound)
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta_min = beta_min
        self.gamma_val = gamma_val
        self.fireflies = self.initialize_fireflies()
        self.light_intensity = np.zeros(n_fireflies)
        self.best_firefly = None
        self.best_intensity = float("inf")
        self.intensity_history = []

    def initialize_fireflies(self):
        kmeans = KMeans(n_clusters=self.n_fireflies)
        clusters = kmeans.fit(np.random.uniform(self.lower_bound, self.upper_bound, (self.n_fireflies * 2, self.n_dim)))
        return clusters.cluster_centers_

    def update_light_intensity(self):
        for i in range(self.n_fireflies):
            self.light_intensity[i] = objective_function(self.fireflies[i])
            if self.light_intensity[i] < self.best_intensity:
                self.best_intensity = self.light_intensity[i]
                self.best_firefly = self.fireflies[i]

    def attractiveness(self, r):
        return self.beta_min * np.exp(-self.gamma_val * r ** 2)

    def move_fireflies(self, iteration):
        for i in range(self.n_fireflies):
            for j in range(self.n_fireflies):
                if self.light_intensity[i] > self.light_intensity[j]:
                    r = np.linalg.norm(self.fireflies[i] - self.fireflies[j])
                    beta = self.attractiveness(r)
                    step_size = (1 - r / self.upper_bound)
                    self.fireflies[i] = (
                        self.fireflies[i] * (1 - beta) +
                        self.fireflies[j] * beta +
                        step_size * (np.random.rand(self.n_dim) - 0.5)
                    )
                    self.fireflies[i] = np.clip(self.fireflies[i], self.lower_bound, self.upper_bound)

    def optimize(self):
        start_time = time.time()
        for t in range(self.max_iter):
            self.update_light_intensity()
            self.move_fireflies(t)
            self.intensity_history.append(self.best_intensity)
            print(f"Iteration {t+1}/{self.max_iter}: Best Intensity = {self.best_intensity:.6e}")
        end_time = time.time()
        execution_time_sec = end_time - start_time
        print(f"Execution Time (seconds): {execution_time_sec}")
        return self.best_firefly, self.best_intensity

# Parameters
n_dim = len(hospitals)
lower_bound = np.zeros(n_dim)
upper_bound = max_capacity
n_fireflies = 20
max_iter = 100
alpha = 0.1
beta_min = 0.2
gamma_val = 1.0

# Run optimization
fa = FireflyAlgorithm(n_fireflies, n_dim, lower_bound, upper_bound, max_iter, alpha, beta_min, gamma_val)
best_solution, best_intensity = fa.optimize()
best_solution = np.round(best_solution)

# Normalize firefly positions
firefly_positions = np.array(fa.fireflies)
normalized_positions = (firefly_positions - np.min(firefly_positions, axis=0)) / (
    np.ptp(firefly_positions, axis=0) + 1e-6
)

# Apply PCA
pca = PCA(n_components=2)
pca_results = pca.fit_transform(normalized_positions)
plt.figure(figsize=(8, 6))
plt.scatter(pca_results[:, 0], pca_results[:, 1], c='blue', edgecolor='black', s=80)
plt.title("PCA: Firefly Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid()
plt.show()

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=10, random_state=42)
tsne_results = tsne.fit_transform(normalized_positions)
plt.figure(figsize=(8, 6))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c='orange', edgecolor='black', s=80)
plt.title("t-SNE: Firefly Clustering")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.grid()
plt.show()
