#Enhanced Algorithm with K-Means, Levy Flight

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.special import gamma as gamma_func  # Import gamma as gamma_func
import time

def objective_function(x): #Sample Objective Function Formula for sum of squared elements
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
        self.light_intensity = np.zeros(n_fireflies)
        self.best_firefly = None
        self.best_intensity = float("inf")

    def initialize_fireflies(self): #K-Means Clustering function for Firely Initialization
        kmeans = KMeans(n_clusters=self.n_fireflies) 
        clusters = kmeans.fit(np.random.uniform(self.lower_bound, self.upper_bound, (self.n_fireflies * 2, self.n_dim)))
        return clusters.cluster_centers_

    def update_light_intensity(self):
        for i in range(self.n_fireflies):
            self.light_intensity[i] = objective_function(self.fireflies[i])
            if self.light_intensity[i] < self.best_intensity:
                self.best_intensity = self.light_intensity[i]
                self.best_firefly = self.fireflies[i]

    def attractiveness(self, r): #iteration parameter
        #dynamic_light_absorption = 1.0 / (1 + (iteration / self.max_iter)) 
        return self.beta_min * np.exp(-self.gamma_val * r**2) #* dynamic_light_absorption

    def levy_flight(self, Lambda): #Levy Flight Function
        sigma = (gamma_func(1 + Lambda) * np.sin(np.pi * Lambda / 2) / (gamma_func((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2))) ** (1 / Lambda)
        u = np.random.randn() * sigma
        v = np.random.randn()
        step = u / abs(v)**(1 / Lambda)
        return step

    def move_fireflies(self, iteration): #Adaptive Randomization Parameter:: Allow the randomization parameter 𝛼 to adapt based on the fireflies' interactions and positions.
        #dynamic_alpha = self.alpha * (1 - iteration/self.max_iter )
        for i in range(self.n_fireflies):
            for j in range(self.n_fireflies):
                if self.light_intensity[i] > self.light_intensity[j]:
                    r = np.linalg.norm(self.fireflies[i] - self.fireflies[j])
                    beta = self.attractiveness(r)
                    L = self.levy_flight(1.5)  # Levy flight Formula
                    step_size = (1 - r / self.upper_bound) #* dynamic_alpha  # Dynamic alpha step size
                    self.fireflies[i] = (
                        self.fireflies[i] * (1 - beta) +
                        self.fireflies[j] * beta +
                        step_size * (np.random.rand(self.n_dim) - 0.5) +
                        L
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
upper_bound =10
n_fireflies = 20
max_iter = 100
alpha = 0.1
beta_min = 0.4
gamma_val = 1.0

# Initialize and run the algorithm
fa = FireflyAlgorithm(n_fireflies, n_dim, lower_bound, upper_bound, max_iter, alpha, beta_min, gamma_val)
best_solution, best_intensity = fa.optimize()

print("SOP 1 and 2")
print("Best solution:", best_solution)
print(f"Best objective value: {best_intensity:.6f}")

# Calculate distances from each firefly to the best solution
#distances = np.linalg.norm(fa.fireflies - best_solution, axis=1)
#threshold = 5  #Adjust this threshold as needed

#close_fireflies = np.sum(distances < threshold)
#print(f"Number of fireflies close to the best solution (within {threshold} distance):", close_fireflies)

'''plt.scatter(fa.fireflies[:, 0], fa.fireflies[:, 1], c='yellow', label='Fireflies')
plt.scatter(fa.best_firefly[0], fa.best_firefly[1], c='red', label='Best Solution')
plt.title(f'Iteration {max_iter}')
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show() '''

# Output is a 1 because of the objective value. N square is expected to have an output of 1 or
