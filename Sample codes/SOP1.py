import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time

def objective_function(x):  # Sample Objective Function Formula for sum of squared elements
    return np.sum(x**2)

class FireflyAlgorithm:
    def __init__(self, n_fireflies, n_dim, lower_bound, upper_bound, max_iter, alpha=0.5, beta_min=0.2, gamma_val=1.0):
        self.n_fireflies = n_fireflies
        self.n_dim = n_dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta_min = beta_min
        self.gamma_val = gamma_val

        # Initialize fireflies in clusters
        self.fireflies = self.initialize_fireflies()
        print("\n🔹 Initialized Fireflies via K-Means:")
        for idx, f in enumerate(self.fireflies):
            print(f"Firefly {idx + 1}: {np.round(f, 2)}")
        self.light_intensity = np.zeros(n_fireflies)
        self.best_firefly = None
        self.best_intensity = float("inf")

    def initialize_fireflies(self):  # K-Means Clustering function for Firefly Initialization
        kmeans = KMeans(n_clusters=self.n_fireflies)
        clusters = kmeans.fit(np.random.uniform(self.lower_bound, self.upper_bound, (self.n_fireflies * 2, self.n_dim)))
        return clusters.cluster_centers_

    def update_light_intensity(self):
        for i in range(self.n_fireflies):
            self.light_intensity[i] = objective_function(self.fireflies[i])
            if self.light_intensity[i] < self.best_intensity:
                self.best_intensity = self.light_intensity[i]
                self.best_firefly = self.fireflies[i]

    def attractiveness(self, r):  # Iteration parameter
        return self.beta_min * np.exp(-self.gamma_val * r**2)

    def move_fireflies(self, iteration):  # Adaptive Randomization Parameter
        dynamic_alpha = self.alpha * (1 - iteration / self.max_iter)
        for i in range(self.n_fireflies):
            for j in range(self.n_fireflies):
                if self.light_intensity[i] > self.light_intensity[j]:
                    r = np.linalg.norm(self.fireflies[i] - self.fireflies[j])
                    beta = self.attractiveness(r)
                    step_size = (1 - r / self.upper_bound) * dynamic_alpha  # Dynamic alpha step size
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
            print(f"Iteration {t+1}/{self.max_iter}: Best Intensity = {self.best_intensity:.6f}")
        end_time = time.time()
        execution_time_min = (end_time - start_time) / 60
        execution_time_sec = end_time - start_time
        print(f"Speed Record in Minutes: {execution_time_min}")
        print(f"Speed Record in Seconds: {execution_time_sec}")
        return self.best_firefly, self.best_intensity

# Parameters
n_dim = 2
lower_bound = -10
upper_bound = 10
n_fireflies = 20
max_iter = 100
alpha = 0.1
beta_min = 0.4
gamma_val = 1.0

# Initialize and run the algorithm
fa = FireflyAlgorithm(n_fireflies, n_dim, lower_bound, upper_bound, max_iter, alpha, beta_min, gamma_val)
best_solution, best_intensity = fa.optimize()

print("Best solution:", best_solution)
print(f"Best objective value: {best_intensity:.6f}")

