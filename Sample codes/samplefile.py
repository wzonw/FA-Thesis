import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.special import gamma as gamma_func

def objective_function(x):
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
        self.fireflies = self.initialize_fireflies()
        self.light_intensity = np.zeros(n_fireflies)
        self.best_firefly = None
        self.best_intensity = float("inf")

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
        return self.beta_min * np.exp(-self.gamma_val * r**2)

    def levy_flight(self, Lambda):
        sigma = (gamma_func(1 + Lambda) * np.sin(np.pi * Lambda / 2) / (gamma_func((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2))) ** (1 / Lambda)
        u = np.random.randn() * sigma
        v = np.random.randn()
        step = u / abs(v)**(1 / Lambda)
        return step

    def move_fireflies(self):
        for i in range(self.n_fireflies):
            for j in range(self.n_fireflies):
                if self.light_intensity[i] > self.light_intensity[j]:
                    r = np.linalg.norm(self.fireflies[i] - self.fireflies[j])
                    beta = self.attractiveness(r)
                    L = self.levy_flight(1.5)  # Lévy flight parameter
                    step_size = (1 - r / self.upper_bound) * self.alpha  # Dynamic step size
                    self.fireflies[i] = (
                        self.fireflies[i] * (1 - beta) +
                        self.fireflies[j] * beta +
                        step_size * (np.random.rand(self.n_dim) - 0.5) +
                        L
                    )
                    self.fireflies[i] = np.clip(self.fireflies[i], self.lower_bound, self.upper_bound)

    def optimize(self):
        for t in range(self.max_iter):
            self.update_light_intensity()
            self.move_fireflies()
        return self.best_firefly, self.best_intensity

# Parameters
n_dim = 2
lower_bound = -10
upper_bound = 10
n_fireflies = 20
max_iter = 100
alpha = 0.1
beta_min = 0.2
gamma_val = 1.0

# Run multiple times to ensure consistency
n_runs = 10
best_solutions = []
best_intensities = []

for _ in range(n_runs):
    fa = FireflyAlgorithm(n_fireflies, n_dim, lower_bound, upper_bound, max_iter, alpha, beta_min, gamma_val)
    best_solution, best_intensity = fa.optimize()
    best_solutions.append(best_solution)
    best_intensities.append(best_intensity)

# Statistical Analysis
mean_best_intensity = np.mean(best_intensities)
std_best_intensity = np.std(best_intensities)

print("Mean best objective value:", mean_best_intensity)
print("Standard deviation of best objective values:", std_best_intensity)

# Visualization
# Set limits for the axes to provide a normal perspective
x_min, x_max = -10, 10
y_min, y_max = -10, 10

plt.figure(figsize=(8, 6))
for best_solution in best_solutions:
    plt.scatter(best_solution[0], best_solution[1], c='blue', label='Best Solution')

# Remove duplicate labels in the legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title('Best Solutions Found Over Multiple Runs')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True)
plt.show()

