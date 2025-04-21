import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import gamma as gamma_func

def objective_function(x): # Sample Objective Function Formula for sum of squared elements
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

        # Initialize fireflies randomly
        self.fireflies = np.random.uniform(self.lower_bound, self.upper_bound, (self.n_fireflies, self.n_dim))
        self.light_intensity = np.zeros(n_fireflies)
        self.best_firefly = None
        self.best_intensity = float("inf")

    def update_light_intensity(self):
        for i in range(self.n_fireflies):
            self.light_intensity[i] = objective_function(self.fireflies[i])
            if self.light_intensity[i] < self.best_intensity:
                self.best_intensity = self.light_intensity[i]
                self.best_firefly = self.fireflies[i]

    def attractiveness(self, r, beta, gamma): #iteration parameter
        return beta * np.exp(-self.gamma_val * r**2)

    def move_fireflies(self, iteration): 
        # Dynamic parameter adjustment
        dynamic_alpha = self.alpha * (1 - iteration / self.max_iter)  # Adjust alpha for exploration-exploitation
        dynamic_beta = self.beta_min + (1 - self.beta_min) * np.exp(-iteration / (self.max_iter / 2))  # Beta decay for convergence
        dynamic_gamma = self.gamma_val * (1 - iteration / self.max_iter)  # Gamma decay to balance attraction
        
        for i in range(self.n_fireflies):
            for j in range(self.n_fireflies):
                if self.light_intensity[i] > self.light_intensity[j]:
                    r = np.linalg.norm(self.fireflies[i] - self.fireflies[j])
                    beta = self.attractiveness(r, dynamic_gamma, dynamic_beta)
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
            print(f"iteration {t+1}/{self.max_iter}: Best Intensity = {self.best_intensity:.6f} ")
        end_time = time.time()
        execution_time_min = (end_time - start_time)/60
        execution_time_sec = end_time - start_time
        print(f"Speed Record in Minute/s: {execution_time_min}")
        print(f"Speed Record in Second/s: {execution_time_sec}")    
        return self.best_firefly, self.best_intensity

# Parameters
n_dim = 2
lower_bound = -10
upper_bound = 10
n_fireflies = 20
max_iter = 15
alpha = 0.1
beta_min = 0.2
gamma_val = 1.0

# Initialize and run the algorithm
fa = FireflyAlgorithm(n_fireflies, n_dim, lower_bound, upper_bound, max_iter, alpha, beta_min, gamma_val)
best_solution, best_intensity = fa.optimize()

print("SOP 3")
print("Best solution:", best_solution)
print(f"Best objective value: ", best_intensity)

alphas, betas, gammas = [], [], []

for t in range(max_iter):
    dynamic_alpha = alpha * (1 - t / max_iter)
    dynamic_beta = beta_min + (1 - beta_min) * np.exp(-t / (max_iter / 2))
    dynamic_gamma = gamma_val * (1 - t / max_iter)

    alphas.append(dynamic_alpha)
    betas.append(dynamic_beta)
    gammas.append(dynamic_gamma)

plt.figure(figsize=(10, 5))
plt.plot(alphas, label="Alpha")
plt.plot(betas, label="Beta")
plt.plot(gammas, label="Gamma")
plt.title("Dynamic Parameter Adjustment Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()
